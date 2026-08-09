"""PP-OCR text detection and recognition inference."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from shapely import BufferJoinStyle
from shapely.geometry import Polygon

from .model_loader import (
    MODEL_REVISION,
    PPOCR_DET_MODEL_FILENAME,
    PPOCR_DICT_FILENAME,
    PPOCR_REC_MODEL_FILENAME,
    download_model_file,
)

_CHARACTER_FILTER_NAMES = (
    "areas",
    "hiragana",
    "alphabet",
    "numbers",
    "symbols",
)


@dataclass(slots=True, eq=False)
class OCRResult:
    """1枚の画像に対するOCR結果。"""

    image: np.ndarray = field(repr=False)
    boxes: list[list[list[int]]]  # 各文字領域の4頂点座標
    scores: list[float]  # 各認識結果の信頼度
    texts: list[str]  # 認識結果の文字列

    def __post_init__(self) -> None:
        _validate_bgr_image(self.image)
        lengths = {len(self.boxes), len(self.scores), len(self.texts)}
        if len(lengths) != 1:
            raise ValueError("OCR result fields must have the same length")


class CTCDecoder:
    FILTER_NAMES = _CHARACTER_FILTER_NAMES

    def __init__(
        self,
        dict_path: str | Path,
        characters_path: str | Path = "characters.yml",
        *,
        new_area_names: list[str] | None = None,
    ):
        """キャラクター辞書の読み込み
        Args:
            dict_path: PaddleOCR公式の辞書ファイルへのパス
            characters_path: ナンバープレートで使用する文字を分類したYAMLへのパス
            new_area_names: YAMLの地名に追加する新しい地名
        """
        with Path(dict_path).open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        try:
            dictionary = data["PostProcess"]["character_dict"]
        except (KeyError, TypeError) as error:
            raise ValueError("invalid PP-OCR character dictionary") from error
        if not isinstance(dictionary, list) or not dictionary:
            raise ValueError("PP-OCR character dictionary must be a non-empty list")

        # PaddleOCR reserves index zero for CTC blank and appends a space.
        self.character = ["<blank>", *(str(value) for value in dictionary), " "]

        with Path(characters_path).open("r", encoding="utf-8") as file:
            filter_characters = yaml.safe_load(file)
        character_array = np.asarray(self.character)
        self.filter_masks = {}
        for filter_name in self.FILTER_NAMES:
            try:
                values = filter_characters[filter_name]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"character filter is missing {filter_name!r}"
                ) from error
            if not isinstance(values, list):
                raise ValueError(f"character filter {filter_name!r} must be a list")
            if filter_name == "areas" and new_area_names:
                values = [*values, *new_area_names]
            # 地域名のような複数文字の値も、OCR辞書の単位に合わせて1文字ずつ展開する
            allowed_characters = {
                character for value in values for character in str(value)
            }
            self.filter_masks[filter_name] = np.isin(
                character_array, list(allowed_characters)
            )

    def decode(
        self,
        preds: np.ndarray,
        filters: Iterable[str] | None = FILTER_NAMES,
    ) -> tuple[str, float]:
        """ONNXモデルの出力（確率マップ）を文字列に変換する
        Args:
            preds: ONNX Runtimeの出力 (Shape: [1, タイムステップ数, 辞書数])
            filters: 使用する文字分類（areas, hiragana, alphabet, numbers, symbols）。
                Noneの場合はフィルターしない。
        """
        probabilities = np.asarray(preds)
        if probabilities.ndim != 3 or probabilities.shape[0] != 1:
            raise ValueError("preds must have shape (1, timesteps, characters)")
        if probabilities.shape[1] == 0:
            raise ValueError("preds must contain at least one timestep")
        if probabilities.shape[2] != len(self.character):
            raise ValueError("preds character count does not match the OCR dictionary")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("preds must contain only finite values")

        if filters is not None:
            if isinstance(filters, str):
                raise TypeError("filters must be an iterable of filter names")
            filters = tuple(filters)
            invalid_filters = set(filters) - set(self.FILTER_NAMES)
            if invalid_filters:
                names = ", ".join(sorted(invalid_filters))
                raise ValueError(f"Unknown character filters: {names}")

            # 複数フィルターは和集合として扱い、CTCのBlankは常に候補に残す
            character_mask = np.zeros(len(self.character), dtype=bool)
            for filter_name in filters:
                character_mask |= self.filter_masks[filter_name]
            character_mask[0] = True
            probabilities = np.where(
                character_mask[None, None, :], probabilities, -np.inf
            )

        # 各タイムステップで最も確率が高い文字の「インデックス」と「その確率」を取得
        # preds[0] -> [タイムステップ数, 辞書数]
        preds_idx = np.argmax(probabilities[0], axis=1)
        preds_prob = np.max(probabilities[0], axis=1)
        char_list = []
        conf_list = []
        # 前のタイムステップのインデックスを記憶する変数（重複チェック用）
        prev_idx = 0
        # CTCデコードのメインループ
        for idx, prob in zip(preds_idx, preds_prob):
            # ルール1: Blankトークン（インデックス0）は無視する
            if idx == 0:
                prev_idx = idx
                continue
            # ルール2: 直前の文字と同じインデックスが連続した場合は無視する
            if idx == prev_idx:
                prev_idx = idx
                continue
            # 条件をクリアした文字を辞書から引いて結合
            char_list.append(self.character[idx])
            conf_list.append(prob)
            prev_idx = idx
        # 文字列に結合
        text = "".join(char_list)
        # テキスト全体の信頼度スコア（平均確率）を計算
        score = np.mean(conf_list) if len(conf_list) > 0 else 0.0
        return text, float(score)


def _validate_bgr_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image with shape (H, W, 3)")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")
    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")


def _validate_probability(name: str, value: float) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _validate_positive_integer(name: str, value: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def preprocess_det(
    img: np.ndarray, target_size: int = 736
) -> tuple[np.ndarray, tuple[float, float]]:
    """PP-OCRv6 テキスト検出モデル向け前処理
    Args:
        img: OpenCVで読み込んだBGR画像 (H, W, C)
        target_size: 長辺の目標ピクセルサイズ（32の倍数）
    """
    _validate_bgr_image(img)
    target_size = _validate_positive_integer("target_size", target_size)
    h, w, _ = img.shape
    # 1. 縦横比を維持しながら、長辺を目標サイズに合わせる（かつ32の倍数に変形）
    scale = target_size / max(h, w)
    new_h = int(round(h * scale / 32) * 32)
    new_w = int(round(w * scale / 32) * 32)
    # 最低でも32ピクセル以上を確保
    new_h = max(32, new_h)
    new_w = max(32, new_w)
    img_resized = cv2.resize(img, (new_w, new_h))
    # 2. BGRからRGBへ色空間を変換
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # 3. 0.0〜1.0 に正規化
    img_data = img_rgb.astype(np.float32) / 255.0
    # 4. ImageNetの平均・標準偏差で正規化 (PP-OCR共通仕様)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    # 5. HWC (Height, Width, Channel) -> NCHW (Batch, Channel, Height, Width)
    img_data = np.transpose(img_data, (2, 0, 1))  # (C, H, W)
    img_data = np.expand_dims(img_data, axis=0)  # (1, C, H, W)
    return np.ascontiguousarray(img_data), (new_w / w, new_h / h)


def postprocess_dbnet(
    onnx_outputs: list[np.ndarray],
    original_shape: tuple[int, int],
    scale: float | tuple[float, float],
    thresh: float = 0.3,
    box_thresh: float = 0.6,
    max_candidates: int = 1000,
) -> tuple[list[list[list[int]]], list[float]]:
    """DBNetのONNX出力から元の画像サイズの4頂点座標リストへ変換する
    Args:
        onnx_outputs: ONNX Runtimeの session.run() から返された出力リスト (通常は shape [1, 1, H, W])
        original_shape: 元画像のサイズ (元のH, 元のW)
        scale: 前処理（リサイズ）時に適用した拡大縮小倍率 (ターゲットサイズ / 元の長辺)
        thresh: 確率マップを2値化するための閾値
        box_thresh: 抽出したテキストボックス全体の平均スコアに対する閾値
        max_candidates: 検出するテキストボックスの最大数
    """
    if len(original_shape) != 2 or min(original_shape) <= 0:
        raise ValueError("original_shape must contain positive height and width")
    thresh = _validate_probability("thresh", thresh)
    box_thresh = _validate_probability("box_thresh", box_thresh)
    max_candidates = _validate_positive_integer("max_candidates", max_candidates)
    if isinstance(scale, Real):
        scale_x = scale_y = float(scale)
    else:
        if len(scale) != 2:
            raise ValueError("scale must be a number or an (x, y) pair")
        scale_x, scale_y = (float(value) for value in scale)
    if (
        not np.isfinite(scale_x)
        or not np.isfinite(scale_y)
        or scale_x <= 0.0
        or scale_y <= 0.0
    ):
        raise ValueError("scale values must be finite and greater than zero")
    if not onnx_outputs:
        raise ValueError("onnx_outputs must not be empty")

    pred = np.asarray(onnx_outputs[0])
    if pred.ndim == 4 and pred.shape[:2] == (1, 1):
        pred = pred[0, 0]
    elif pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    elif pred.ndim != 2:
        raise ValueError("DBNet output must have shape (1, 1, H, W) or (H, W)")
    if pred.size == 0 or not np.all(np.isfinite(pred)):
        raise ValueError("DBNet output must be non-empty and finite")
    # 2. 閾値(thresh)を基に2値化（テキスト領域を1、背景を0にする）
    segmentation = (pred > thresh).astype(np.uint8)
    # 3. OpenCVを使って輪郭（Contours）を抽出
    contours, _ = cv2.findContours(segmentation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    scores = []
    num_contours = min(len(contours), max_candidates)
    for index in range(num_contours):
        contour = contours[index]

        # 4. 輪郭を囲む最小の傾いた四角形（minAreaRect）を取得
        # ※極端に小さいノイズ領域は無視する
        box, min_side = get_mini_boxes(contour)
        if min_side < 3:
            continue
        # 5. ボックス内の平均確率（スコア）を計算し、信頼度の低いものを除外
        score = box_score_fast(pred, box.astype(np.int32))
        if box_thresh > score:
            continue
        # 6. 四角形をPaddleOCRの仕様に合わせて少しだけ外側に広げる（文字の切れ端を救う処理）
        box = unclip(box, unclip_ratio=1.5)
        if len(box) < 3:
            continue
        box, min_side = get_mini_boxes(box)
        if min_side < 3:
            continue
        # 7. 前処理で変形・縮小された座標を、元の画像サイズ（スケール）に復元
        # x, y それぞれをリサイズ倍率(scale)で割る
        box[:, 0] = np.clip(box[:, 0] / scale_x, 0, original_shape[1] - 1)
        box[:, 1] = np.clip(box[:, 1] / scale_y, 0, original_shape[0] - 1)
        boxes.append(box.astype(np.int32).tolist())
        scores.append(float(score))
    return boxes, scores


def get_mini_boxes(contour: np.ndarray) -> tuple[np.ndarray, float]:
    """輪郭から最小の外接矩形（4頂点）と短辺の長さを取得する補助関数"""
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    # 頂点の順序を（左上、右上、右下、左下）にソ整列
    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2
    box = np.array([points[index_1], points[index_2], points[index_3], points[index_4]])
    min_side = min(bounding_box[1][0], bounding_box[1][1])
    return box, min_side


def box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
    """四角形領域内の確率マップの平均値を高速に計算する補助関数"""
    h, w = bitmap.shape[:2]
    box = box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, [box.astype(np.int32)], 1)

    return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]


def unclip(box: np.ndarray, unclip_ratio: float = 1.5) -> np.ndarray:
    """テキストボックスを少しだけ外側に拡張する補助関数（ポリゴン拡張）"""
    poly = Polygon(box)
    if not poly.is_valid:
        poly = poly.convex_hull
    if poly.is_empty or poly.length <= 0.0:
        return np.empty((0, 2), dtype=np.float32)
    distance = poly.area * unclip_ratio / poly.length
    offset = poly.buffer(distance, join_style=BufferJoinStyle.mitre)
    if offset.is_empty:
        return np.empty((0, 2), dtype=np.float32)
    if offset.geom_type == "MultiPolygon":
        offset = max(offset.geoms, key=lambda geometry: geometry.area)
    if not hasattr(offset, "exterior"):
        return np.empty((0, 2), dtype=np.float32)
    src_pt = np.array(offset.exterior.coords)
    # shapelyの仕様上、最初と最後の頂点が重複(クローズドパス)するため、最後の1点を削る
    if len(src_pt) > 1 and np.array_equal(src_pt[0], src_pt[-1]):
        src_pt = src_pt[:-1]

    # OpenCV の後段処理で扱える (N, 2) 配列にして返す。
    return src_pt.astype(np.float32)


def crop_and_get_perspective(
    img: np.ndarray, points: list[list[int]] | np.ndarray
) -> np.ndarray:
    """斜めのテキスト領域を水平に補正して切り出す（透視変換）
    Args:
        img: 元のBGR画像 (H, W, C)
        points: 検出された4つの頂点座標。形状は (4, 2) の numpy 配列
                [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
                （通常、左上・右上・右下・左下の順）
    """
    _validate_bgr_image(img)
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("points must have shape (4, 2)")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must contain only finite values")
    # points は左上、右上、右下、左下の順で受け取る。
    rect = pts
    if abs(float(cv2.contourArea(rect))) <= 1e-6:
        raise ValueError("points must describe a non-degenerate quadrilateral")
    (tl, tr, br, bl) = rect
    # 2. 切り出し後の新しい画像（水平）の幅（Width）を計算
    # 上辺の長さと下辺の長さの最大値を採用
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(round(width_a)), int(round(width_b)))
    # 3. 切り出し後の新しい画像（水平）の高さ（Height）を計算
    # 右辺の長さと左辺の長さの最大値を採用
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(round(height_a)), int(round(height_b)))
    if max_width < 1 or max_height < 1:
        raise ValueError("points must describe a non-degenerate quadrilateral")
    # 4. 変換先の水平な座標（ターゲット）を定義
    dst = np.array(
        [
            [0, 0],  # 左上
            [max_width - 1, 0],  # 右上
            [max_width - 1, max_height - 1],  # 右下
            [0, max_height - 1],  # 左下
        ],
        dtype=np.float32,
    )
    # 5. 透視変換マトリクスを計算し、画像を歪み補正して切り出し
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped_img = cv2.warpPerspective(img, transform_matrix, (max_width, max_height))
    # 6. PaddleOCRの仕様に合わせる（縦幅が横幅より極端に長い場合、文字が縦書き or 180度回転していると判断して補正）
    # ※通常の横書きテキストであればこのステップはスルーされます
    if max_height * 1.0 / max_width >= 2.0:
        warped_img = cv2.rotate(warped_img, cv2.ROTATE_90_CLOCKWISE)
    return warped_img


def preprocess_rec(cropped_img: np.ndarray, target_height: int = 48) -> np.ndarray:
    """PP-OCRv6 テキスト認識モデル向け前処理
    Args:
        cropped_img: 検出結果のバウンディングボックスから切り出した、傾き補正済みの部分画像(BGR)
        target_height: PP-OCRv6では 48 ピクセル固定
    """
    _validate_bgr_image(cropped_img)
    target_height = _validate_positive_integer("target_height", target_height)
    img_h, img_w, _ = cropped_img.shape
    # 1. 縦幅を48pxに固定し、横幅をアスペクト比を維持してリサイズ
    scale = target_height / img_h
    new_w = max(1, int(round(img_w * scale)))
    # 非常に長いテキスト用に最大幅を制限（必要に応じて調整。一般的には制限なし可変か320/640等）
    # new_w = min(new_w, 320)
    img_resized = cv2.resize(cropped_img, (new_w, target_height))
    # 2. BGRからRGBへ色空間を変換
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # 3. 画素値を -1.0 〜 1.0 の範囲に正規化する標準スケーリング
    img_data = img_rgb.astype(np.float32) / 255.0
    img_data = (img_data - 0.5) / 0.5
    # 4. HWC -> NCHW へ次元を変換
    img_data = np.transpose(img_data, (2, 0, 1))  # (C, H, W)
    img_data = np.expand_dims(img_data, axis=0)  # (1, C, H, W)
    return np.ascontiguousarray(img_data)


class PPOCR:
    """PP-OCRの文字検出・文字認識をまとめて実行する。"""

    def __init__(
        self,
        det_model_path: str | Path | None = None,
        rec_model_path: str | Path | None = None,
        dict_path: str | Path | None = None,
        characters_path: str | Path | None = None,
        *,
        new_area_names: list[str] | None = None,
        det_target_size: int = 736,
        rec_target_height: int = 48,
        det_thresh: float = 0.3,
        det_box_thresh: float = 0.6,
        max_candidates: int = 1000,
        character_filters: Iterable[str] | None = CTCDecoder.FILTER_NAMES,
        providers: list[str] | None = None,
        cache_dir: str | Path | None = None,
        revision: str = MODEL_REVISION,
        local_files_only: bool = False,
    ):
        self.det_target_size = _validate_positive_integer(
            "det_target_size", det_target_size
        )
        self.rec_target_height = _validate_positive_integer(
            "rec_target_height", rec_target_height
        )
        self.det_thresh = _validate_probability("det_thresh", det_thresh)
        self.det_box_thresh = _validate_probability("det_box_thresh", det_box_thresh)
        self.max_candidates = _validate_positive_integer(
            "max_candidates", max_candidates
        )
        if isinstance(character_filters, str):
            raise TypeError("character_filters must be an iterable of filter names")
        self.character_filters = (
            None if character_filters is None else list(character_filters)
        )
        if self.character_filters is not None:
            unknown_filters = set(self.character_filters).difference(
                _CHARACTER_FILTER_NAMES
            )
            if unknown_filters:
                names = ", ".join(sorted(unknown_filters))
                raise ValueError(f"Unknown character filters: {names}")

        def resolve_model_file(path: str | Path | None, filename: str) -> Path:
            if path is not None:
                return Path(path)
            return download_model_file(
                filename,
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=local_files_only,
            )

        det_model_path = resolve_model_file(det_model_path, PPOCR_DET_MODEL_FILENAME)
        rec_model_path = resolve_model_file(rec_model_path, PPOCR_REC_MODEL_FILENAME)
        dict_path = resolve_model_file(dict_path, PPOCR_DICT_FILENAME)
        if characters_path is None:
            characters_path = (
                Path(__file__).resolve().parents[1] / "configs" / "characters.yml"
            )

        session_kwargs = {} if providers is None else {"providers": providers}
        self.det_session = ort.InferenceSession(str(det_model_path), **session_kwargs)
        self.rec_session = ort.InferenceSession(str(rec_model_path), **session_kwargs)
        det_inputs = self.det_session.get_inputs()
        rec_inputs = self.rec_session.get_inputs()
        if not det_inputs or not rec_inputs:
            raise RuntimeError("PP-OCR models must each have an input")
        self.det_input_name = det_inputs[0].name
        self.rec_input_name = rec_inputs[0].name
        self.decoder = CTCDecoder(
            dict_path, characters_path, new_area_names=new_area_names
        )

    def __call__(self, image: np.ndarray) -> OCRResult:
        """BGR画像を受け取り、検出領域・信頼度・認識文字列を返す。"""
        input_tensor, scale = self._preprocess(image)
        det_outputs = self.det_session.run(None, {self.det_input_name: input_tensor})
        return self._postprocess(image, det_outputs, scale)

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """入力画像を検証し、文字検出モデル用テンソルへ変換する。"""
        return preprocess_det(image, target_size=self.det_target_size)

    def _postprocess(
        self,
        image: np.ndarray,
        det_outputs: list[np.ndarray],
        scale: tuple[float, float],
    ) -> OCRResult:
        """検出出力を座標に戻し、各領域を文字認識して結果を構築する。"""
        boxes, _ = postprocess_dbnet(
            det_outputs,
            original_shape=image.shape[:2],
            scale=scale,
            thresh=self.det_thresh,
            box_thresh=self.det_box_thresh,
            max_candidates=self.max_candidates,
        )

        texts = []
        scores = []
        for box in boxes:
            cropped_text = crop_and_get_perspective(image, box)
            rec_input = preprocess_rec(
                cropped_text, target_height=self.rec_target_height
            )
            rec_outputs = self.rec_session.run(None, {self.rec_input_name: rec_input})
            text, score = self.decoder.decode(
                rec_outputs[0], filters=self.character_filters
            )
            texts.append(text)
            scores.append(score)

        return OCRResult(image=image, boxes=boxes, scores=scores, texts=texts)
