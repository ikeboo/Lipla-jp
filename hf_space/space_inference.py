"""Gradio UIから独立したナンバープレート認識処理。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import fields
from functools import cache
from typing import Any

import numpy as np
import spaces

from lipla import LPDetResult, Recognizer

_IMAGE_FIELD_NAMES = frozenset(
    {"plate_image", "original_image", "_det_image", "_result_image"}
)


@cache
def get_recognizer() -> Recognizer:
    """モデルを最初の推論時に一度だけ初期化する。"""
    return Recognizer(providers=["CPUExecutionProvider"])


def _json_compatible(value: Any) -> Any:
    """NumPyの値をJSONで表現できるPython組み込み型へ変換する。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    return value


def result_to_dict(result: LPDetResult) -> dict[str, Any]:
    """LPDetResultから画像フィールドを除いたJSON用データを作る。"""
    return {
        field.name: _json_compatible(getattr(result, field.name))
        for field in fields(result)
        if field.name not in _IMAGE_FIELD_NAMES and not field.name.startswith("_")
    }


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """OpenCVのBGR画像をGradio表示用RGB画像へ変換する。"""
    return np.ascontiguousarray(image[..., ::-1])


@spaces.GPU(duration=60)
def recognize_image(
    image: np.ndarray | None,
    *,
    recognizer_factory: Callable[[], Recognizer] = get_recognizer,
) -> tuple[
    list[tuple[np.ndarray, str]],
    list[tuple[np.ndarray, str]],
    str,
]:
    """RGB画像を認識し、2種類の画像ギャラリーとJSON文字列を返す。"""
    if image is None:
        return [], [], "[]"
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")

    bgr_image = _bgr_to_rgb(image)
    results = recognizer_factory()(bgr_image)
    det_images = [
        (_bgr_to_rgb(result.det_image), f"LPDetResult[{index}]")
        for index, result in enumerate(results)
    ]
    result_images = [
        (_bgr_to_rgb(result.result_image), f"LPDetResult[{index}]")
        for index, result in enumerate(results)
    ]
    result_json = json.dumps(
        [result_to_dict(result) for result in results],
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    return det_images, result_images, result_json
