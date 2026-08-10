"""ナンバープレート認識で使用する画像の検証と前処理。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

LOCAL_BLACK_VALUE_MAX = 50
LOCAL_GREEN_HUE_MIN = 30
LOCAL_GREEN_HUE_MAX = 95
LOCAL_GREEN_SATURATION_MIN = 25
LOCAL_GREEN_VALUE_MAX = 130


def validate_bgr_image(image: np.ndarray) -> None:
    """BGR画像として利用できる配列か検証する。

    Args:
        image: 検証対象の画像。

    Raises:
        TypeError: ``image`` が ``numpy.ndarray`` ではない場合、または
            データ型が ``uint8`` ではない場合。
        ValueError: ``image`` が空、または形状が ``(H, W, 3)`` ではない場合。
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image with shape (H, W, 3)")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")
    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")


def read_bgr_image(path: str | Path) -> np.ndarray:
    """ファイルパスからBGR画像を読み込む。

    OpenCVのパス処理に依存せずバイト列を復号するため、日本語を含むパスも
    使用できる。

    Args:
        path: 読み込む画像ファイルのパス。

    Returns:
        ``uint8`` のBGR画像。

    Raises:
        OSError: ファイルを読み込めない場合。
        ValueError: ファイルを画像として復号できない場合。
    """
    image_path = Path(path)
    encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size > 0 else None
    if image is None:
        raise ValueError(f"Could not decode image file: {image_path}")
    return image


def as_bgr(image: np.ndarray) -> np.ndarray:
    """正規化後の画像をBGR形式へ揃える。

    Args:
        image: グレースケール画像、またはBGR画像。

    Returns:
        BGR形式の画像。入力がBGRの場合は同じ配列を返す。

    Raises:
        TypeError: 入力が配列ではない場合、またはデータ型が ``uint8`` では
            ない場合。
        ValueError: 入力が空、または対応していない形状の場合。
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("normalized image must be a numpy.ndarray")
    if image.ndim == 2:
        if image.size == 0:
            raise ValueError("normalized image must not be empty")
        if image.dtype != np.uint8:
            raise TypeError("normalized image must have dtype uint8")
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        validate_bgr_image(image)
        return image
    raise ValueError(f"normalized image has an invalid shape: {image.shape}")


def preprocess_local_plate(image: np.ndarray) -> np.ndarray:
    """図柄入りナンバープレートから文字色以外を白くする。

    黒色から濃緑色までの画素を残し、OCRを妨げる背景の図柄を除去する。

    Args:
        image: ``uint8`` のBGR画像。

    Returns:
        背景の図柄を白色に置換したBGR画像。

    Raises:
        TypeError: 画像の型またはデータ型が不正な場合。
        ValueError: 画像が空、またはBGR画像ではない場合。
    """
    validate_bgr_image(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    black = value <= LOCAL_BLACK_VALUE_MAX
    dark_green = (
        (hue >= LOCAL_GREEN_HUE_MIN)
        & (hue <= LOCAL_GREEN_HUE_MAX)
        & (saturation >= LOCAL_GREEN_SATURATION_MIN)
        & (value <= LOCAL_GREEN_VALUE_MAX)
    )
    filtered = np.full_like(image, 255)
    keep = black | dark_green
    filtered[keep] = image[keep]
    return filtered


def pad_for_ocr(image: np.ndarray) -> tuple[np.ndarray, int]:
    """OCRの文字検出用に画像の周囲へ余白を追加する。

    Args:
        image: 余白を追加するBGR画像。

    Returns:
        余白を追加した画像と、上下左右に追加したピクセル数。

    Raises:
        TypeError: 画像の型またはデータ型が不正な場合。
        ValueError: 画像が空、またはBGR画像ではない場合。
    """
    validate_bgr_image(image)
    padding = max(8, int(round(image.shape[0] * 0.10)))
    padded = cv2.copyMakeBorder(
        image,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=(127, 127, 127),
    )
    return padded, padding


def remove_padding(image: np.ndarray, padding: int) -> np.ndarray:
    """画像から上下左右の余白を取り除く。

    余白がゼロ以下、または画像に対して大きすぎる場合は入力をそのまま返す。

    Args:
        image: 余白を含む画像。
        padding: 上下左右から取り除くピクセル数。

    Returns:
        余白を取り除いた画像、または元の画像。
    """
    if padding <= 0:
        return image
    if image.shape[0] <= padding * 2 or image.shape[1] <= padding * 2:
        return image
    return image[padding:-padding, padding:-padding]
