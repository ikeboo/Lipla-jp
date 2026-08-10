"""ナンバープレートの射影・色正規化。"""

import cv2
import numpy as np


class PlateNormalizer:
    """ナンバープレートを指定サイズへ射影・色補正する。

    Args:
        width: 出力画像の幅。
        height: 出力画像の高さ。

    Raises:
        TypeError: 幅または高さが整数ではない場合。
        ValueError: 幅または高さが2未満の場合。
    """

    def __init__(self, width: int = 320, height: int = 160):
        for name, value in (("width", width), ("height", height)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 2:
                raise ValueError(f"{name} must be at least 2")
        self.width = width
        self.height = height

    def normalize(self, image: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        """4頂点で囲まれたナンバープレートを正規化する。

        Args:
            image: プレートを含むBGR画像。
            vertices: 左上、右上、右下、左下の順に並べた4頂点座標。

        Returns:
            指定サイズへ射影・色補正したBGR画像。

        Raises:
            TypeError: 画像または頂点の型が不正な場合。
            ValueError: 画像、頂点の形状、または四角形が不正な場合。
        """
        self._validate_image(image)
        source_points = np.asarray(vertices, dtype=np.float32)
        if source_points.shape != (4, 2):
            raise ValueError("vertices must have shape (4, 2)")
        if not np.all(np.isfinite(source_points)):
            raise ValueError("vertices must contain only finite values")
        if abs(float(cv2.contourArea(source_points))) <= 1e-6:
            raise ValueError("vertices must describe a non-degenerate quadrilateral")
        if not cv2.isContourConvex(source_points):
            raise ValueError("vertices must describe a convex quadrilateral")

        dst_points = np.array(
            [
                [0, 0],
                [self.width - 1, 0],
                [self.width - 1, self.height - 1],
                [0, self.height - 1],
            ],
            dtype=np.float32,
        )

        # 射影変換行列を計算
        transform = cv2.getPerspectiveTransform(source_points, dst_points)

        # 射影変換を適用して正規化された画像を取得
        warped_image = cv2.warpPerspective(image, transform, (self.width, self.height))
        normalized_image = self.normalize_color(warped_image)

        return normalized_image

    def normalize_color(self, image: np.ndarray) -> np.ndarray:
        """ナンバープレート画像の輝度コントラストを補正する。

        Args:
            image: 補正するBGR画像。

        Returns:
            必要に応じてCLAHEを適用したBGR画像。

        Raises:
            TypeError: 画像の型またはデータ型が不正な場合。
            ValueError: 画像が空、またはBGR画像ではない場合。
        """
        self._validate_image(image)

        # OCRモデルは3チャンネル入力で学習されているため、グレースケールへ
        # 潰さない。元々十分なコントラストがある画像は色を保ち、低コントラ
        # スト時だけ輝度を局所補正する。常時のヒストグラム均一化は、大きな
        # かな文字を背景へ溶け込ませることがある。
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab_image)
        low, high = np.percentile(lightness, (5, 95))
        if high - low >= 45:
            return np.ascontiguousarray(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 4))
        lightness = clahe.apply(lightness)
        normalized_lab = cv2.merge((lightness, channel_a, channel_b))
        return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a BGR image with shape (H, W, 3)")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("image must not be empty")
        if image.dtype != np.uint8:
            raise TypeError("image must have dtype uint8")
