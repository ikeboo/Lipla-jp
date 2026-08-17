"""ONNX Runtime execution provider selection helpers."""

from __future__ import annotations

import importlib
import threading
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import onnxruntime as ort

_WEBGPU_REGISTRATION_NAME = "lipla_webgpu_ep"
_WEBGPU_LOCK = threading.Lock()
_WEBGPU_OPTIONS = {
    "preferredLayout": "NHWC",
    "powerPreference": "high-performance",
}


@lru_cache(maxsize=1)
def _webgpu_devices() -> tuple[Any, ...]:
    """Register the optional WebGPU plugin and return usable devices."""
    try:
        webgpu_ep = importlib.import_module("onnxruntime_ep_webgpu")
    except ImportError:
        return ()

    required_apis = (
        "register_execution_provider_library",
        "get_ep_devices",
    )
    if not all(hasattr(ort, name) for name in required_apis):
        warnings.warn(
            "onnxruntime-ep-webgpu is installed, but this onnxruntime version "
            "does not provide the plugin EP API; falling back to CPUExecutionProvider",
            RuntimeWarning,
            stacklevel=2,
        )
        return ()

    try:
        with _WEBGPU_LOCK:
            try:
                ort.register_execution_provider_library(
                    _WEBGPU_REGISTRATION_NAME,
                    webgpu_ep.get_library_path(),
                )
            except RuntimeError as exc:
                # Another library may already have registered this plugin globally.
                if "already registered" not in str(exc).lower():
                    raise

            ep_name = webgpu_ep.get_ep_name()
            return tuple(
                device for device in ort.get_ep_devices() if device.ep_name == ep_name
            )
    except Exception as exc:  # pragma: no cover - depends on driver/runtime state
        warnings.warn(
            f"WebGPU Execution Provider could not be initialized ({exc}); "
            "falling back to CPUExecutionProvider",
            RuntimeWarning,
            stacklevel=2,
        )
        return ()


def create_inference_session(
    model_path: str | Path,
    *,
    providers: list[str] | None = None,
    session_options: ort.SessionOptions | None = None,
) -> ort.InferenceSession:
    """Create a session, preferring the optional WebGPU plugin when available.

    An explicit ``providers`` list takes precedence over automatic selection.
    Without one, WebGPU is selected when the plugin can discover a device. ONNX
    Runtime keeps its built-in CPU EP as the fallback for unsupported nodes and
    for environments where no WebGPU device is found.
    """
    options = session_options or ort.SessionOptions()
    if providers is not None:
        return ort.InferenceSession(
            str(model_path), sess_options=options, providers=providers
        )

    devices = _webgpu_devices()
    if devices:
        options.add_provider_for_devices(devices, _WEBGPU_OPTIONS)
    return ort.InferenceSession(str(model_path), sess_options=options)
