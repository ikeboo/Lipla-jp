"""OCR出力をナンバープレートの各項目へ変換する解析処理。"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np

from lipla.inferencers.ppocr import OCRResult

from .image_processing import validate_bgr_image

FIELD_NAMES: Final = ("area", "class_number", "kana", "number")
CandidateMap: TypeAlias = dict[str, list["TextCandidate"]]

_ALNUM_RE = re.compile(r"[^0-9A-Z]")
_DIGIT_RE = re.compile(r"\d")
_CLASS_SUFFIX_CHARACTERS = frozenset("0123456789ACFHKLMPXY")


@dataclass(slots=True, frozen=True)
class TextCandidate:
    """OCRが生成した文字列候補。

    Attributes:
        text: 認識された文字列。
        score: 文字列の認識信頼度。
    """

    text: str
    score: float


@dataclass(slots=True, frozen=True)
class ParsedPlateFields:
    """ナンバープレートの4項目から選択した候補。

    Attributes:
        area: 使用の本拠の位置を表す地名の候補。
        class_number: 分類番号の候補。
        kana: 用途を表すひらがなの候補。
        number: 一連指定番号の候補。
    """

    area: TextCandidate
    class_number: TextCandidate
    kana: TextCandidate
    number: TextCandidate


def create_candidate_map() -> CandidateMap:
    """各プレート項目に空の候補リストを持つ辞書を作成する。

    Returns:
        地名、分類番号、ひらがな、一連指定番号の候補辞書。
    """
    return {field_name: [] for field_name in FIELD_NAMES}


def edit_distance(left: str, right: str) -> int:
    """2つの文字列間のレーベンシュタイン距離を計算する。

    Args:
        left: 比較する1つ目の文字列。
        right: 比較する2つ目の文字列。

    Returns:
        挿入、削除、置換による最小編集回数。
    """
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, start=1):
        current = [row]
        for column, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


class OCRResultParser:
    """空間OCR結果と固定領域OCR候補をプレート項目へ整理する。

    Args:
        areas: 認識結果として許可する地名。
        kana_characters: 認識結果として許可するひらがな。

    Raises:
        ValueError: ``areas`` が空の場合。
    """

    def __init__(self, areas: Iterable[str], kana_characters: Iterable[str]) -> None:
        self.areas = tuple(dict.fromkeys(map(str, areas)))
        if not self.areas:
            raise ValueError("areas must not be empty")
        self.kana_characters = tuple(dict.fromkeys(map(str, kana_characters)))
        self.hiragana = frozenset(self.kana_characters)

    def parse(
        self,
        ocr_result: OCRResult,
        *,
        field_candidates: Mapping[str, Iterable[TextCandidate]] | None = None,
        padding: int = 0,
        spatial_fields: Iterable[str] | None = None,
    ) -> ParsedPlateFields:
        """OCR結果からプレートの4項目を選択する。

        Args:
            ocr_result: プレート全体に対する空間OCR結果。
            field_candidates: 固定領域OCRなどが生成した項目別候補。
            padding: OCR画像の上下左右に追加済みの余白サイズ。
            spatial_fields: 空間OCR候補を利用する項目。``None`` の場合は
                全項目で利用する。

        Returns:
            項目ごとに妥当性と信頼度から選択した候補。

        Raises:
            TypeError: ``padding`` が整数ではない場合、またはOCR画像の型が
                不正な場合。
            ValueError: ``padding`` や項目名が不正な場合、またはOCR画像が
                BGR画像ではない場合。
        """
        self._validate_input(ocr_result, padding)
        candidates = create_candidate_map()
        if field_candidates:
            self._extend_candidates(candidates, field_candidates)

        if spatial_fields is None:
            self.collect_spatial_candidates(ocr_result, candidates, padding=padding)
        else:
            selected_fields = frozenset(spatial_fields)
            self._validate_field_names(selected_fields, "spatial")
            spatial_candidates = create_candidate_map()
            self.collect_spatial_candidates(
                ocr_result, spatial_candidates, padding=padding
            )
            for field_name in selected_fields:
                candidates[field_name].extend(spatial_candidates[field_name])

        return ParsedPlateFields(
            area=self.best_area(candidates["area"]),
            class_number=self.best_class_number(candidates["class_number"]),
            kana=self.best_kana(candidates["kana"]),
            number=self.best_number(candidates["number"]),
        )

    def collect_spatial_candidates(
        self,
        ocr_result: OCRResult,
        candidates: MutableMapping[str, list[TextCandidate]],
        *,
        padding: int,
    ) -> None:
        """文字領域の位置から各プレート項目の候補を収集する。

        Args:
            ocr_result: 文字領域の座標を含むOCR結果。
            candidates: 候補を追加する項目別辞書。
            padding: OCR画像の上下左右に追加済みの余白サイズ。
        """
        image_height, image_width = ocr_result.image.shape[:2]
        plate_height = max(1, image_height - padding * 2)
        plate_width = max(1, image_width - padding * 2)

        for box, text, score in zip(
            ocr_result.boxes, ocr_result.texts, ocr_result.scores
        ):
            text = str(text).strip()
            if not text:
                continue
            try:
                points = np.asarray(box, dtype=np.float32).reshape(4, 2)
            except (TypeError, ValueError):
                continue
            if not np.all(np.isfinite(points)):
                continue

            center_x = (float(np.mean(points[:, 0])) - padding) / plate_width
            center_y = (float(np.mean(points[:, 1])) - padding) / plate_height
            candidate = TextCandidate(text, float(score))
            if center_y < 0.48:
                field_name = "area" if center_x < 0.52 else "class_number"
                candidates[field_name].append(candidate)
                self.split_top_text(candidate, candidates)
            else:
                self.split_bottom_text(candidate, candidates)

    def split_top_text(
        self,
        candidate: TextCandidate,
        candidates: MutableMapping[str, list[TextCandidate]],
    ) -> None:
        """上段の結合文字列を地名と分類番号へ分割する。

        Args:
            candidate: 上段のOCR候補。
            candidates: 分割した候補を追加する項目別辞書。
        """
        compact = re.sub(r"\s+", "", candidate.text).upper()
        match = re.search(r"([0-9A-Z]{2,3})$", compact)
        if match:
            area_text = compact[: match.start()]
            if area_text:
                candidates["area"].append(TextCandidate(area_text, candidate.score))
            candidates["class_number"].append(
                TextCandidate(match.group(1), candidate.score)
            )
            return

        for area in self.areas:
            if compact.startswith(area):
                candidates["area"].append(TextCandidate(area, candidate.score))
                suffix = compact[len(area) :]
                if suffix:
                    candidates["class_number"].append(
                        TextCandidate(suffix, candidate.score)
                    )
                return

    def split_bottom_text(
        self,
        candidate: TextCandidate,
        candidates: MutableMapping[str, list[TextCandidate]],
    ) -> None:
        """下段の結合文字列をひらがなと一連指定番号へ分割する。

        Args:
            candidate: 下段のOCR候補。
            candidates: 分割した候補を追加する項目別辞書。
        """
        kana = next(
            (character for character in candidate.text if character in self.hiragana),
            "",
        )
        if kana:
            candidates["kana"].append(TextCandidate(kana, candidate.score))
        if any(character.isdigit() for character in candidate.text):
            candidates["number"].append(candidate)

    def best_area(self, candidates: Iterable[TextCandidate]) -> TextCandidate:
        """地名候補から語彙に最も近い有効な候補を選ぶ。

        Args:
            candidates: 地名の文字列候補。

        Returns:
            編集距離で補正した最良候補。候補がない場合は空文字列。
        """
        best = TextCandidate("", 0.0)
        for candidate in candidates:
            text = re.sub(r"[\s・·\-0-9A-Z]", "", candidate.text.upper())
            if not text:
                continue
            if text in self.areas:
                mapped = text
                similarity = 1.0
            else:
                mapped, distance = min(
                    ((area, edit_distance(text, area)) for area in self.areas),
                    key=lambda item: (item[1], abs(len(item[0]) - len(text))),
                )
                similarity = max(0.0, 1.0 - distance / max(len(text), len(mapped)))
                if similarity < 0.45:
                    continue
            score = float(candidate.score) * similarity
            if score > best.score:
                best = TextCandidate(mapped, score)
        return best

    def best_class_number(self, candidates: Iterable[TextCandidate]) -> TextCandidate:
        """候補から書式が妥当な分類番号を選ぶ。

        Args:
            candidates: 分類番号の文字列候補。

        Returns:
            信頼度が最大の有効候補。候補がない場合は空文字列。
        """
        valid_candidates: list[TextCandidate] = []
        for candidate in candidates:
            text = _ALNUM_RE.sub("", candidate.text.upper())
            if len(text) not in (2, 3):
                continue
            if not text[0].isdigit() or not text[1].isdigit():
                continue
            if len(text) == 3 and text[2] not in _CLASS_SUFFIX_CHARACTERS:
                continue
            valid_candidates.append(TextCandidate(text, float(candidate.score)))
        if not valid_candidates:
            return TextCandidate("", 0.0)
        return max(valid_candidates, key=lambda value: value.score)

    def best_kana(self, candidates: Iterable[TextCandidate]) -> TextCandidate:
        """候補から有効なひらがなを1文字選ぶ。

        Args:
            candidates: ひらがなの文字列候補。

        Returns:
            信頼度が最大の有効候補。候補がない場合は空文字列。
        """
        best = TextCandidate("", 0.0)
        for candidate in candidates:
            text = next(
                (
                    character
                    for character in candidate.text
                    if character in self.hiragana
                ),
                "",
            )
            if text and candidate.score > best.score:
                best = TextCandidate(text, float(candidate.score))
        return best

    def best_number(self, candidates: Iterable[TextCandidate]) -> TextCandidate:
        """候補から1から9999までの一連指定番号を選ぶ。

        Args:
            candidates: 一連指定番号の文字列候補。

        Returns:
            信頼度が最大の有効候補。候補がない場合は空文字列。
        """
        best = TextCandidate("", 0.0)
        for candidate in candidates:
            digits = "".join(_DIGIT_RE.findall(candidate.text))
            if not 1 <= len(digits) <= 4:
                continue
            value = int(digits)
            if not 1 <= value <= 9999:
                continue
            if candidate.score > best.score:
                best = TextCandidate(str(value), float(candidate.score))
        return best

    @staticmethod
    def _validate_input(ocr_result: OCRResult, padding: int) -> None:
        if not isinstance(padding, int) or isinstance(padding, bool):
            raise TypeError("padding must be an integer")
        if padding < 0:
            raise ValueError("padding must not be negative")
        validate_bgr_image(ocr_result.image)
        if padding and padding * 2 >= min(ocr_result.image.shape[:2]):
            raise ValueError("padding is too large for the OCR image")

    @classmethod
    def _validate_field_names(cls, field_names: Iterable[str], label: str) -> None:
        unknown_fields = set(field_names).difference(FIELD_NAMES)
        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Unknown {label} fields: {names}")

    @classmethod
    def _extend_candidates(
        cls,
        candidates: CandidateMap,
        additions: Mapping[str, Iterable[TextCandidate]],
    ) -> None:
        cls._validate_field_names(additions, "candidate")
        for field_name, values in additions.items():
            candidates[field_name].extend(values)
