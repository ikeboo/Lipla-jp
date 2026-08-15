"""日本のナンバープレートを検出・認識する公開インターフェース。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import yaml
from PIL import ImageFont

from lipla.inferencers.ec_pose import ECPose
from lipla.inferencers.model_loader import MODEL_REVISION
from lipla.inferencers.ppocr import PPOCR, OCRResult

from .fixed_field_recognizer import (
    FixedFieldRecognizer,
    ctc_log_probability,
    logsumexp,
)
from .image_processing import (
    as_bgr,
    pad_for_ocr,
    preprocess_local_plate,
    read_bgr_image,
    remove_padding,
    validate_bgr_image,
)
from .ocr_result_parser import (
    CandidateMap,
    OCRResultParser,
    TextCandidate,
    edit_distance,
)
from .plate_normalizer import PlateNormalizer
from .pose_postprocessor import (
    PoseDetection,
    polygon_iou,
    suppress_duplicate_detections,
)
from .recognition_result import (
    SYSTEM_JAPANESE_FONT_CANDIDATES,
    LPDetResult,
    render_result_image,
)

_SYSTEM_JAPANESE_FONT_CANDIDATES = SYSTEM_JAPANESE_FONT_CANDIDATES
_TextCandidate = TextCandidate


def _load_system_japanese_font(font_size: int) -> ImageFont.FreeTypeFont:
    """OSにインストール済みの日本語対応フォントを読み込む。

    従来の内部APIとの互換性を保つため、このモジュール上のフォント候補と
    ``ImageFont`` を参照する。

    Args:
        font_size: 読み込むフォントサイズ。

    Returns:
        最初に読み込みに成功したフォント。

    Raises:
        OSError: 候補のフォントを1つも読み込めなかった場合。
    """
    last_error: OSError | None = None
    for font_name in _SYSTEM_JAPANESE_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError as error:
            last_error = error

    candidates = ", ".join(_SYSTEM_JAPANESE_FONT_CANDIDATES)
    raise OSError(
        "No Japanese-capable system font was found. "
        f"Install a Japanese font such as Noto Sans CJK. Tried: {candidates}"
    ) from last_error


def _validate_bgr_image(image: np.ndarray) -> None:
    """BGR画像の形式を検証する。

    Args:
        image: 検証対象の画像。
    """
    validate_bgr_image(image)


def _create_result_image(result: LPDetResult) -> np.ndarray:
    """プレートと認識文字列を1枚のBGR画像へ描画する。

    Args:
        result: 描画するナンバープレート認識結果。

    Returns:
        認識内容を描画したBGR画像。
    """
    return render_result_image(result, font_loader=_load_system_japanese_font)


# 旧モジュールのフォント差し替えをLPDetResultにも反映する互換設定。
LPDetResult._font_loader = staticmethod(
    lambda font_size: _load_system_japanese_font(font_size)
)


class Recognizer:
    """BGR画像内にある日本のナンバープレートを検出・認識する。

    姿勢推定、射影補正、OCR、候補解析を独立したコンポーネントへ委譲し、
    このクラスは処理順序の制御と公開APIの提供だけを担当する。

    Args:
        pose_model_path: プレート検出用EdgeCrafterPoseONNXモデルのパス。``None`` の場合は既定の
            モデルをダウンロードする。
        ocr_det_model_path: PPOCR検出ONNXモデルのパス。
        ocr_rec_model_path: PPOCR認識ONNXモデルのパス。
        det_thresh: プレート検出の信頼度しきい値。
        ocr_dict_path: PPOCR文字辞書のパス。
        new_area_names: 新規地名のリスト。(新規地名が発行された際に利用)
        providers: ONNX Runtimeの実行プロバイダー。
        cache_dir: モデルのキャッシュディレクトリ。
        revision: モデルリポジトリのリビジョン。
        local_files_only: ローカルキャッシュだけを利用するか。
    """

    _LOCAL_SPATIAL_FIELDS = frozenset(("area", "class_number", "kana"))

    def __init__(
        self,
        pose_model_path: str | Path | None = None,
        ocr_det_model_path: str | Path | None = None,
        ocr_rec_model_path: str | Path | None = None,
        *,
        det_thresh: float = 0.7,
        ocr_dict_path: str | Path | None = None,
        new_area_names: list[str] | None = None,
        providers: list[str] | None = None,
        cache_dir: str | Path | None = None,
        revision: str = MODEL_REVISION,
        local_files_only: bool = False,
    ) -> None:
        characters_path = (
            Path(__file__).resolve().parents[1] / "configs" / "characters.yml"
        )
        self.pose_model = ECPose(
            pose_model_path,
            thresh=det_thresh,
            providers=providers,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
        )
        self.plate_normalizer = PlateNormalizer()
        self.ocr_model = PPOCR(
            ocr_det_model_path,
            ocr_rec_model_path,
            dict_path=ocr_dict_path,
            characters_path=characters_path,
            new_area_names=new_area_names,
            providers=providers,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
        )

        with characters_path.open("r", encoding="utf-8") as file:
            characters = yaml.safe_load(file)
        area_names = [*characters["areas"], *(new_area_names or ())]
        self.areas = tuple(dict.fromkeys(map(str, area_names)))
        self.kana_characters = tuple(
            dict.fromkeys(str(value) for value in characters["hiragana"])
        )
        self.hiragana = frozenset(self.kana_characters)

        self.ocr_parser = OCRResultParser(self.areas, self.kana_characters)
        self.field_recognizer = FixedFieldRecognizer(self.ocr_model, self.ocr_parser)
        self._area_tokens = self.field_recognizer.area_tokens

    def __call__(self, image: np.ndarray | str | Path) -> list[LPDetResult]:
        """画像内の全ナンバープレートを認識する。

        Args:
            image: ``(H, W, 3)`` 形状のOpenCV BGR画像、または画像ファイルの
                パス。日本語を含むパスも利用できる。

        Returns:
            検出信頼度順の認識結果。プレートがない場合は空リスト。

        Raises:
            TypeError: 入力画像の型またはデータ型が不正な場合。
            ValueError: 入力画像が空、またはBGR画像ではない場合。
            RuntimeError: 姿勢推定結果の各フィールド数が一致しない場合。
        """
        if isinstance(image, (str, Path)):
            image = read_bgr_image(image)
        validate_bgr_image(image)
        pose_result = self.pose_model(image)

        lengths = {
            len(pose_result.kpts),
            len(pose_result.scores),
            len(pose_result.class_names),
        }
        if len(lengths) != 1:
            raise RuntimeError("pose result fields must have the same length")

        detections = list(
            zip(
                pose_result.kpts,
                pose_result.scores,
                pose_result.class_names,
            )
        )
        results: list[LPDetResult] = []
        for raw_vertices, det_score, class_name in suppress_duplicate_detections(
            detections
        ):
            result = self._recognize_detection(
                image, raw_vertices, det_score, class_name
            )
            if result is not None:
                results.append(result)
        return results

    def _recognize_detection(
        self,
        image: np.ndarray,
        raw_vertices: list[float],
        detection_score: float,
        class_name: str,
    ) -> LPDetResult | None:
        vertices = np.asarray(raw_vertices, dtype=np.float32).reshape(-1, 2)
        if vertices.shape != (4, 2) or not np.all(np.isfinite(vertices)):
            return None

        # ECPoseは左上・左下・右下・右上、正規化器は左上・右上・右下・左下。
        normalizer_vertices = np.ascontiguousarray(
            vertices[[0, 3, 2, 1]], dtype=np.float32
        )
        normalized = self.plate_normalizer.normalize(image, normalizer_vertices)
        normalized = as_bgr(normalized)

        class_name = str(class_name)
        is_local = class_name.casefold() == "local"
        ocr_image = preprocess_local_plate(normalized) if is_local else normalized
        padded, padding = pad_for_ocr(ocr_image)
        ocr_result = self.ocr_model(padded)
        field_candidates = self.field_recognizer.recognize(normalized)
        result = self.parse_ocr_result(
            ocr_result,
            vertices=vertices,
            detection_score=float(detection_score),
            normalized_image=normalized,
            original_image=image,
            class_name=class_name,
            field_candidates=field_candidates,
            padding=padding,
            spatial_fields=(self._LOCAL_SPATIAL_FIELDS if is_local else None),
        )
        if result.class_number and result.number > 0:
            return result
        return None

    def parse_ocr_result(
        self,
        ocr_result: OCRResult,
        *,
        vertices: np.ndarray | None = None,
        detection_score: float = 0.0,
        normalized_image: np.ndarray | None = None,
        original_image: np.ndarray | None = None,
        class_name: str = "",
        field_candidates: Mapping[str, Iterable[TextCandidate]] | None = None,
        padding: int = 0,
        spatial_fields: Iterable[str] | None = None,
    ) -> LPDetResult:
        """OCR結果を4項目へ解析して認識結果を構築する。

        画像や頂点に関するキーワード引数を省略できるため、OCR結果だけを
        個別に確認する用途にも利用できる。

        Args:
            ocr_result: プレート全体に対する空間OCR結果。
            vertices: 元画像上の4頂点座標。
            detection_score: プレート検出の信頼度。
            normalized_image: 射影・色補正済みのプレート画像。
            original_image: プレートを検出した元画像。
            class_name: 検出モデルが返したプレート種別。
            field_candidates: 固定領域OCRなどが生成した項目別候補。
            padding: OCR画像の上下左右に追加済みの余白サイズ。
            spatial_fields: 空間OCR候補を利用する項目。

        Returns:
            画像情報と4項目の認識内容を持つ結果。
        """
        fields = self.ocr_parser.parse(
            ocr_result,
            field_candidates=field_candidates,
            padding=padding,
            spatial_fields=spatial_fields,
        )
        plate_image = (
            normalized_image
            if normalized_image is not None
            else remove_padding(ocr_result.image, padding)
        )
        source_image = original_image if original_image is not None else plate_image
        result_vertices = (
            np.asarray(vertices, dtype=np.float32).reshape(4, 2)
            if vertices is not None
            else np.zeros((4, 2), dtype=np.float32)
        )
        return LPDetResult(
            vertices=result_vertices,
            score=float(detection_score),
            plate_image=plate_image,
            original_image=source_image,
            class_name=class_name,
            area=fields.area.text,
            area_score=fields.area.score,
            class_number=fields.class_number.text,
            class_number_score=fields.class_number.score,
            kana=fields.kana.text,
            kana_score=fields.kana.score,
            number=int(fields.number.text) if fields.number.text else 0,
            number_score=fields.number.score,
        )

    def create_result_images(self, results: list[LPDetResult]) -> list[np.ndarray]:
        """各認識結果の表示用画像を作成する。

        Args:
            results: このインスタンスが返した認識結果。

        Returns:
            入力と同じ順序の表示用BGR画像。
        """
        return [result.result_image for result in results]

    @staticmethod
    def _pose_nms(
        detections: list[PoseDetection], iou_threshold: float = 0.3
    ) -> list[PoseDetection]:
        """重複する姿勢推定結果を除外する。

        Args:
            detections: 頂点、信頼度、クラス名からなる検出結果。
            iou_threshold: 同一物体とみなすIoUの下限。

        Returns:
            重複を除外した検出結果。
        """
        return suppress_duplicate_detections(detections, iou_threshold)

    _polygon_iou = staticmethod(polygon_iou)
    _validate_image = staticmethod(validate_bgr_image)
    _read_image = staticmethod(read_bgr_image)
    _as_bgr = staticmethod(as_bgr)
    _preprocess_local_plate = staticmethod(preprocess_local_plate)
    _pad_for_ocr = staticmethod(pad_for_ocr)
    _remove_padding = staticmethod(remove_padding)
    _edit_distance = staticmethod(edit_distance)
    _logsumexp = staticmethod(logsumexp)
    _ctc_log_probability = staticmethod(ctc_log_probability)

    def _recognize_fixed_fields(self, image: np.ndarray) -> CandidateMap:
        return self.field_recognizer.recognize(image)
