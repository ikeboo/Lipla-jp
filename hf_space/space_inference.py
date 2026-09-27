"""Gradio UIから独立したナンバープレート認識処理。"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import fields
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import spaces

from lipla import Recognizer, Result

_IMAGE_FIELD_NAMES = frozenset(
    {"plate_image", "original_image", "_det_image", "_result_image"}
)


def _log_webgpu_installation() -> None:
    """起動時にWebGPUプラグインのインストール状態をSpaceログへ出す。"""
    try:
        plugin_version = version("onnxruntime-ep-webgpu")
    except PackageNotFoundError:
        plugin_version = "not installed"
    print(
        "[Lipla] onnxruntime-ep-webgpu="
        f"{plugin_version}; device availability is checked in the ZeroGPU worker",
        flush=True,
    )


_log_webgpu_installation()


@cache
def get_recognizer() -> Recognizer:
    """モデルをワーカーごとに一度初期化し、利用中のEPをログへ出す。"""
    started = time.perf_counter()
    print(f"[Lipla] Initializing Recognizer in worker pid={os.getpid()}", flush=True)
    recognizer = Recognizer()
    session_providers = {
        "pose": recognizer.pose_model.session.get_providers(),
        "ocr_det": recognizer.ocr_model.det_session.get_providers(),
        "ocr_rec": recognizer.ocr_model.rec_session.get_providers(),
    }
    webgpu_active = all(
        any(provider.lower() == "webgpuexecutionprovider" for provider in providers)
        for providers in session_providers.values()
    )
    elapsed = time.perf_counter() - started
    print(
        f"[Lipla] Recognizer initialized in {elapsed:.3f}s; "
        f"WebGPU active={webgpu_active}; providers={session_providers}",
        flush=True,
    )
    return recognizer


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


def result_to_dict(result: Result) -> dict[str, Any]:
    """Resultから画像フィールドを除いたJSON用データを作る。"""
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
        (_bgr_to_rgb(result.det_image), f"Result[{index}]")
        for index, result in enumerate(results)
    ]
    result_images = [
        (_bgr_to_rgb(result.result_image), f"Result[{index}]")
        for index, result in enumerate(results)
    ]
    result_json = json.dumps(
        [result_to_dict(result) for result in results],
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    return det_images, result_images, result_json
