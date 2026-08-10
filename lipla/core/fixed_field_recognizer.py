"""ナンバープレートの固定領域に特化した文字認識。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final, Protocol

import numpy as np

from lipla.inferencers.ppocr import preprocess_rec

from .image_processing import validate_bgr_image
from .ocr_result_parser import (
    CandidateMap,
    OCRResultParser,
    TextCandidate,
    create_candidate_map,
)

FIELD_RECTS: Final = {
    "area": (0.194, 0.00, 0.519, 0.325),
    "class_number": (0.51, 0.00, 0.81, 0.36),
    "number": (0.15, 0.34, 1.00, 1.00),
}
KANA_RECTS: Final = (
    ((0.009, 0.438, 0.197, 1.00), 1.0),
    ((0.016, 0.469, 0.197, 1.00), 0.7),
    ((0.000, 0.375, 0.219, 1.00), 0.7),
)
FIELD_FILTERS: Final = {
    "class_number": ["numbers", "alphabet"],
    "number": ["numbers", "symbols"],
}


class RecognitionSession(Protocol):
    """文字認識セッションに必要なインターフェース。"""

    def run(
        self,
        output_names: None,
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]: ...


class RecognitionDecoder(Protocol):
    """文字認識デコーダーに必要なインターフェース。"""

    character: list[str]

    def decode(
        self,
        predictions: np.ndarray,
        filters: Iterable[str] | None = None,
    ) -> tuple[str, float]: ...


class RecognitionModel(Protocol):
    """固定領域OCRに必要なモデルインターフェース。"""

    rec_session: RecognitionSession
    rec_input_name: str
    rec_target_height: int
    decoder: RecognitionDecoder


def logsumexp(values: np.ndarray) -> float:
    """オーバーフローを避けながら指数の和の対数を計算する。

    Args:
        values: 対数空間の値。

    Returns:
        ``log(sum(exp(values)))`` の計算結果。
    """
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return -np.inf
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def ctc_log_probability(
    log_probabilities: np.ndarray, tokens: tuple[int, ...]
) -> float:
    """1つの語彙に対するCTC前向き確率を対数空間で計算する。

    Args:
        log_probabilities: 各時刻・各文字の対数確率。Blankは添字0とする。
        tokens: 評価する文字列の文字インデックス。

    Returns:
        指定した文字列を生成する全CTCパスの対数確率。

    Raises:
        ValueError: ``tokens`` が空、または時間ステップが不足している場合。
    """
    if not tokens:
        raise ValueError("tokens must not be empty")
    if len(log_probabilities) == 0:
        raise ValueError("log_probabilities must contain at least one timestep")

    extended = [0]
    for token in tokens:
        extended.extend((token, 0))

    previous = np.full(len(extended), -np.inf, dtype=np.float64)
    previous[0] = log_probabilities[0, 0]
    previous[1] = log_probabilities[0, extended[1]]
    for timestep in range(1, len(log_probabilities)):
        current = np.full_like(previous, -np.inf)
        for state, token in enumerate(extended):
            incoming = [previous[state]]
            if state > 0:
                incoming.append(previous[state - 1])
            if state > 1 and token != 0 and token != extended[state - 2]:
                incoming.append(previous[state - 2])
            current[state] = logsumexp(np.asarray(incoming))
            current[state] += log_probabilities[timestep, token]
        previous = current
    return logsumexp(previous[-2:])


class FixedFieldRecognizer:
    """プレートの既知レイアウトから4項目を個別認識する。

    Args:
        ocr_model: PP-OCRの文字認識セッションとデコーダーを持つオブジェクト。
        parser: 地名とひらがなの語彙を保持するOCR結果パーサー。
    """

    def __init__(self, ocr_model: RecognitionModel, parser: OCRResultParser) -> None:
        self.ocr_model = ocr_model
        self.parser = parser
        decoder_characters = self.ocr_model.decoder.character
        self.area_tokens = tuple(
            (
                area,
                tuple(decoder_characters.index(character) for character in area),
            )
            for area in parser.areas
        )

    def recognize(self, image: np.ndarray) -> CandidateMap:
        """固定領域を切り出して項目別の文字列候補を生成する。

        必要な文字認識インターフェースをOCRモデルが持たない場合は、全項目が
        空の候補辞書を返す。これにより、文字検出だけを実装した差し替えモデル
        でも空間OCR解析を利用できる。

        Args:
            image: 射影・色補正済みのBGRプレート画像。

        Returns:
            項目別の文字列候補。

        Raises:
            TypeError: 画像の型またはデータ型が不正な場合。
            ValueError: 画像が空、またはBGR画像ではない場合。
        """
        validate_bgr_image(image)
        candidates = create_candidate_map()
        required = (
            "rec_session",
            "rec_input_name",
            "rec_target_height",
            "decoder",
        )
        if not all(hasattr(self.ocr_model, name) for name in required):
            return candidates

        height, width = image.shape[:2]
        for field_name, rect in FIELD_RECTS.items():
            crop = self._crop(image, rect)
            if field_name == "area":
                candidate = self._recognize_area(crop)
            elif field_name == "class_number":
                candidate = self._recognize_class_number(crop)
            else:
                candidate = self._recognize_crop(crop, FIELD_FILTERS[field_name])
            if candidate.text:
                candidates[field_name].append(candidate)

        kana_candidate = self._recognize_kana(image)
        if kana_candidate.text:
            candidates["kana"].append(kana_candidate)

        top_row = image[
            : max(1, int(round(height * 0.41))),
            int(round(width * 0.14)) : int(round(width * 0.85)),
        ]
        self.parser.split_top_text(self._recognize_crop(top_row, None), candidates)
        return candidates

    @staticmethod
    def _crop(image: np.ndarray, rect: tuple[float, float, float, float]) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = rect
        left = max(0, min(width - 1, int(round(x1 * width))))
        top = max(0, min(height - 1, int(round(y1 * height))))
        right = max(left + 1, min(width, int(round(x2 * width))))
        bottom = max(top + 1, min(height, int(round(y2 * height))))
        return image[top:bottom, left:right]

    def _recognize_crop(
        self, crop: np.ndarray, filters: list[str] | None
    ) -> TextCandidate:
        if crop.size == 0:
            return TextCandidate("", 0.0)
        rec_input = preprocess_rec(crop, target_height=self.ocr_model.rec_target_height)
        rec_outputs = self.ocr_model.rec_session.run(
            None, {self.ocr_model.rec_input_name: rec_input}
        )
        text, score = self.ocr_model.decoder.decode(rec_outputs[0], filters=filters)
        return TextCandidate(str(text).strip(), float(score))

    def _recognize_area(self, crop: np.ndarray) -> TextCandidate:
        if crop.size == 0:
            return TextCandidate("", 0.0)
        rec_input = preprocess_rec(crop, target_height=self.ocr_model.rec_target_height)
        rec_outputs = self.ocr_model.rec_session.run(
            None, {self.ocr_model.rec_input_name: rec_input}
        )[0]
        probabilities = np.asarray(rec_outputs, dtype=np.float64)[0]
        log_probabilities = np.log(np.clip(probabilities, 1e-30, 1.0))
        scores = np.asarray(
            [
                ctc_log_probability(log_probabilities, tokens) / len(tokens)
                for _, tokens in self.area_tokens
            ],
            dtype=np.float64,
        )
        best_index = int(np.argmax(scores))
        normalizer = logsumexp(scores)
        confidence = float(np.exp(scores[best_index] - normalizer))
        return TextCandidate(self.area_tokens[best_index][0], confidence)

    def _recognize_class_number(self, crop: np.ndarray) -> TextCandidate:
        if crop.size == 0:
            return TextCandidate("", 0.0)
        rec_input = preprocess_rec(crop, target_height=self.ocr_model.rec_target_height)
        rec_outputs = self.ocr_model.rec_session.run(
            None, {self.ocr_model.rec_input_name: rec_input}
        )[0]
        text, score = self.ocr_model.decoder.decode(
            rec_outputs, filters=FIELD_FILTERS["class_number"]
        )
        return TextCandidate(str(text).strip().upper(), float(score))

    def _single_character_scores(
        self, crop: np.ndarray, allowed_characters: Iterable[str]
    ) -> dict[str, float]:
        if crop.size == 0:
            return {}
        rec_input = preprocess_rec(crop, target_height=self.ocr_model.rec_target_height)
        rec_outputs = self.ocr_model.rec_session.run(
            None, {self.ocr_model.rec_input_name: rec_input}
        )[0]
        probabilities = np.asarray(rec_outputs)[0]
        character_indices = {
            character: self.ocr_model.decoder.character.index(character)
            for character in allowed_characters
            if character in self.ocr_model.decoder.character
        }
        return {
            character: max(0.0, float(np.max(probabilities[:, index])))
            for character, index in character_indices.items()
        }

    def _recognize_kana(self, image: np.ndarray) -> TextCandidate:
        combined = {character: 0.0 for character in self.parser.kana_characters}
        for rect, weight in KANA_RECTS:
            scores = self._single_character_scores(
                self._crop(image, rect), self.parser.kana_characters
            )
            score_sum = sum(scores.values())
            if score_sum <= 0.0:
                continue
            for character, score in scores.items():
                combined[character] += weight * score / score_sum

        if not combined:
            return TextCandidate("", 0.0)
        character = max(combined, key=combined.get)
        score_sum = sum(combined.values())
        confidence = combined[character] / score_sum if score_sum > 0.0 else 0.0
        return TextCandidate(character, float(confidence))
