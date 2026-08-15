"""ナンバープレート認識結果のデータ表現と可視化。"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .image_processing import validate_bgr_image

FALLBACK_JAPANESE_FONT_URL = (
    "https://github.com/googlefonts/zen-marugothic/raw/refs/heads/main/"
    "fonts/ttf/ZenMaruGothic-Medium.ttf"
)
FALLBACK_JAPANESE_FONT_FILENAME = "ZenMaruGothic-Medium.ttf"
_FONT_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
FALLBACK_JAPANESE_FONT_CACHE_DIR = _FONT_CACHE_HOME / "lipla-jp" / "fonts"
_FONT_DOWNLOAD_TIMEOUT_SECONDS = 30

if sys.platform == "darwin":
    SYSTEM_JAPANESE_FONT_CANDIDATES = (
        "Hiragino Sans GB.ttc",
        "ヒラギノ角ゴシック W3.ttc",
        "Arial Unicode.ttf",
    )
elif sys.platform == "win32":
    SYSTEM_JAPANESE_FONT_CANDIDATES = (
        "YuGothR.ttc",
        "meiryo.ttc",
        "msgothic.ttc",
    )
else:
    SYSTEM_JAPANESE_FONT_CANDIDATES = (
        "NotoSansCJK-Regular.ttc",
        "NotoSansCJKjp-Regular.otf",
        "NotoSansJP-Regular.ttf",
        "NotoSansJP-VariableFont_wght.ttf",
        "ipaexg.ttf",
        "ipag.ttf",
        "VL-Gothic-Regular.ttf",
        "TakaoPGothic.ttf",
    )


class _ResultView(Protocol):
    """結果画像の描画に必要な属性を表すプロトコル。"""

    plate_image: np.ndarray
    area: str
    class_number: str
    kana: str
    number: int


def _download_fallback_japanese_font(font_path: Path) -> Path:
    """GitHubからフォールバック用日本語フォントをキャッシュする。"""
    temporary_path: Path | None = None
    try:
        font_path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            FALLBACK_JAPANESE_FONT_URL,
            headers={"User-Agent": "lipla-jp"},
        )
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=font_path.parent,
            prefix=f".{font_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urllib.request.urlopen(
                request, timeout=_FONT_DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                shutil.copyfileobj(response, temporary_file)

        if temporary_path.stat().st_size == 0:
            raise OSError("downloaded font file is empty")
        temporary_path.replace(font_path)
    except OSError as error:
        raise OSError(
            "Failed to download the fallback Japanese font from "
            f"{FALLBACK_JAPANESE_FONT_URL}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return font_path


def _load_fallback_japanese_font(
    font_size: int,
    *,
    font_loader: Callable[[str | Path, int], ImageFont.FreeTypeFont],
) -> ImageFont.FreeTypeFont:
    """キャッシュ済み、またはダウンロードした日本語フォントを読み込む。"""
    font_path = FALLBACK_JAPANESE_FONT_CACHE_DIR / FALLBACK_JAPANESE_FONT_FILENAME
    if font_path.is_file():
        try:
            return font_loader(font_path, font_size)
        except OSError:
            font_path.unlink(missing_ok=True)

    _download_fallback_japanese_font(font_path)
    try:
        return font_loader(font_path, font_size)
    except OSError as error:
        font_path.unlink(missing_ok=True)
        raise OSError(f"Downloaded fallback font is invalid: {font_path}") from error


def load_system_japanese_font(
    font_size: int,
    candidates: Sequence[str] = SYSTEM_JAPANESE_FONT_CANDIDATES,
) -> ImageFont.FreeTypeFont:
    """OSにインストール済みの日本語対応フォントを読み込む。

    Args:
        font_size: 読み込むフォントサイズ。
        candidates: 優先順に並べたフォントファイル名。

    Returns:
        最初に読み込みに成功したフォント。

    Raises:
        OSError: システムフォントを読み込めず、フォールバック用フォントの
            ダウンロードまたは読み込みにも失敗した場合。
    """
    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            pass

    try:
        return _load_fallback_japanese_font(font_size, font_loader=ImageFont.truetype)
    except OSError as error:
        names = ", ".join(candidates)
        raise OSError(
            "No Japanese-capable font could be loaded. "
            f"Tried system fonts: {names}. "
            f"Fallback URL: {FALLBACK_JAPANESE_FONT_URL}"
        ) from error


def render_result_image(
    result: _ResultView,
    *,
    font_loader: Callable[[int], ImageFont.ImageFont] = load_system_japanese_font,
) -> np.ndarray:
    """認識済みプレートと文字列を1枚のBGR画像へ描画する。

    Args:
        result: プレート画像と認識文字列を持つ結果。
        font_loader: フォントサイズを受け取りフォントを返す関数。

    Returns:
        上部にプレート、下部に認識文字列を配置したBGR画像。

    Raises:
        TypeError: プレート画像の型またはデータ型が不正な場合。
        ValueError: プレート画像が空、またはBGR画像ではない場合。
        OSError: 日本語対応フォントを読み込めない場合。
    """
    validate_bgr_image(result.plate_image)
    plate = np.ascontiguousarray(result.plate_image)
    plate_height, plate_width = plate.shape[:2]
    font_size = max(16, int(round(plate_height * 0.20)))
    font = font_loader(font_size)
    text = f"{result.area} {result.class_number}\n{result.kana} {result.number}"

    measuring_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_box = measuring_draw.multiline_textbbox(
        (0, 0), text, font=font, spacing=4, align="center"
    )
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    padding = max(8, int(round(font_size * 0.4)))
    canvas_width = int(np.ceil(max(plate_width, text_width + padding * 2)))
    canvas_height = int(np.ceil(plate_height + text_height + padding * 2))

    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
    plate_rgb = Image.fromarray(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
    canvas.paste(plate_rgb, ((canvas_width - plate_width) // 2, 0))

    draw = ImageDraw.Draw(canvas)
    text_x = (canvas_width - text_width) / 2 - text_box[0]
    text_y = plate_height + padding - text_box[1]
    draw.multiline_text(
        (text_x, text_y),
        text,
        font=font,
        fill="black",
        spacing=4,
        align="center",
    )
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


@dataclass(slots=True, eq=False)
class LPDetResult:
    """検出したナンバープレートと各項目の認識結果。

    ``vertices`` は姿勢推定モデルと同じく、左上、左下、右下、右上の順で
    保持する。``number`` は整数で保持し、認識できなかった場合は実在しない
    番号であるゼロを格納する。

    Attributes:
        vertices: 元画像上の4頂点座標。
        score: ナンバープレート検出の信頼度。
        plate_image: 射影・色補正済みのプレート画像。
        original_image: 検出元のBGR画像。
        class_name: 検出モデルが返したプレート種別。
        area: 使用の本拠の位置を表す地名。
        area_score: 地名の認識信頼度。
        class_number: 分類番号。
        class_number_score: 分類番号の認識信頼度。
        kana: 用途を表すひらがな。
        kana_score: ひらがなの認識信頼度。
        number: 一連指定番号。
        number_score: 一連指定番号の認識信頼度。
    """

    vertices: np.ndarray
    score: float
    plate_image: np.ndarray = field(repr=False)
    original_image: np.ndarray = field(repr=False)
    class_name: str
    area: str
    area_score: float
    class_number: str
    class_number_score: float
    kana: str
    kana_score: float
    number: int
    number_score: float
    _det_image: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _result_image: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _font_loader: ClassVar[Callable[[int], ImageFont.ImageFont]] = (
        load_system_japanese_font
    )

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float32)
        if vertices.shape != (4, 2):
            raise ValueError("vertices must have shape (4, 2)")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices must contain only finite values")
        validate_bgr_image(self.plate_image)
        validate_bgr_image(self.original_image)
        self.vertices = np.ascontiguousarray(vertices)

    @property
    def det_image(self) -> np.ndarray:
        """検出位置をマゼンタ色で囲んだ元画像を取得する。

        Returns:
            検出した四角形を描画したBGR画像。
        """
        if self._det_image is None:
            validate_bgr_image(self.original_image)
            det_image = self.original_image.copy()
            vertices = np.rint(self.vertices).astype(np.int32).reshape(4, 1, 2)
            cv2.polylines(
                det_image,
                [vertices],
                isClosed=True,
                color=(255, 0, 255),
                thickness=2,
            )
            self._det_image = det_image
        return self._det_image

    @property
    def result_image(self) -> np.ndarray:
        """プレートと認識文字列を並べた結果画像を取得する。

        Returns:
            認識内容を描画したBGR画像。
        """
        if self._result_image is None:
            self._result_image = render_result_image(
                self, font_loader=type(self)._font_loader
            )
        return self._result_image

    def visualize(self) -> None:
        """検出画像と認識結果画像を横に並べて表示する。"""
        import matplotlib.pyplot as plt

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(self.det_image[..., ::-1])
        ax1.set_title("Detected License Plate")
        ax1.axis("off")
        ax2.imshow(self.result_image[..., ::-1])
        ax2.set_title("Recognition Result")
        ax2.axis("off")
        plt.show()
