import sys
from types import SimpleNamespace

import pytest

from lipla.inferencers import execution_provider


class _SessionOptions:
    def __init__(self):
        self.providers = []

    def add_provider_for_devices(self, devices, options):
        self.providers.append((devices, options))


def _clear_webgpu_cache():
    execution_provider._webgpu_devices.cache_clear()


def test_uses_webgpu_plugin_when_device_is_available(monkeypatch):
    device = SimpleNamespace(ep_name="WebGpuExecutionProvider")
    plugin = SimpleNamespace(
        get_library_path=lambda: "/plugin/webgpu.so",
        get_ep_name=lambda: "WebGpuExecutionProvider",
    )
    registrations = []
    sessions = []

    monkeypatch.setitem(sys.modules, "onnxruntime_ep_webgpu", plugin)
    monkeypatch.setattr(
        execution_provider.ort,
        "register_execution_provider_library",
        lambda name, path: registrations.append((name, path)),
    )
    monkeypatch.setattr(execution_provider.ort, "get_ep_devices", lambda: [device])
    monkeypatch.setattr(execution_provider.ort, "SessionOptions", _SessionOptions)
    monkeypatch.setattr(
        execution_provider.ort,
        "InferenceSession",
        lambda path, **kwargs: sessions.append((path, kwargs)) or object(),
    )
    _clear_webgpu_cache()

    execution_provider.create_inference_session("model.onnx")

    assert registrations == [("lipla_webgpu_ep", "/plugin/webgpu.so")]
    options = sessions[0][1]["sess_options"]
    assert options.providers == [
        (
            (device,),
            {
                "preferredLayout": "NHWC",
                "powerPreference": "high-performance",
            },
        )
    ]


def test_falls_back_to_default_cpu_when_plugin_is_missing(monkeypatch):
    real_import_module = execution_provider.importlib.import_module

    def import_module(name):
        if name == "onnxruntime_ep_webgpu":
            raise ImportError
        return real_import_module(name)

    sessions = []
    monkeypatch.setattr(execution_provider.importlib, "import_module", import_module)
    monkeypatch.setattr(execution_provider.ort, "SessionOptions", _SessionOptions)
    monkeypatch.setattr(
        execution_provider.ort,
        "InferenceSession",
        lambda path, **kwargs: sessions.append((path, kwargs)) or object(),
    )
    _clear_webgpu_cache()

    execution_provider.create_inference_session("model.onnx")

    assert sessions[0][1]["sess_options"].providers == []
    assert "providers" not in sessions[0][1]


def test_falls_back_to_default_cpu_when_webgpu_device_is_unavailable(monkeypatch):
    plugin = SimpleNamespace(
        get_library_path=lambda: "/plugin/webgpu.so",
        get_ep_name=lambda: "WebGpuExecutionProvider",
    )
    cpu_device = SimpleNamespace(ep_name="CPUExecutionProvider")
    sessions = []

    monkeypatch.setitem(sys.modules, "onnxruntime_ep_webgpu", plugin)
    monkeypatch.setattr(
        execution_provider.ort,
        "register_execution_provider_library",
        lambda _name, _path: None,
    )
    monkeypatch.setattr(execution_provider.ort, "get_ep_devices", lambda: [cpu_device])
    monkeypatch.setattr(execution_provider.ort, "SessionOptions", _SessionOptions)
    monkeypatch.setattr(
        execution_provider.ort,
        "InferenceSession",
        lambda path, **kwargs: sessions.append((path, kwargs)) or object(),
    )
    _clear_webgpu_cache()

    execution_provider.create_inference_session("model.onnx")

    assert sessions[0][1]["sess_options"].providers == []


def test_explicit_providers_override_automatic_selection(monkeypatch):
    sessions = []
    monkeypatch.setattr(execution_provider.ort, "SessionOptions", _SessionOptions)
    monkeypatch.setattr(
        execution_provider.ort,
        "InferenceSession",
        lambda path, **kwargs: sessions.append((path, kwargs)) or object(),
    )

    execution_provider.create_inference_session(
        "model.onnx", providers=["CPUExecutionProvider"]
    )

    assert sessions[0][1]["providers"] == ["CPUExecutionProvider"]
    assert sessions[0][1]["sess_options"].providers == []


def test_webgpu_retries_fused_activation_failure_without_conv_fusion(monkeypatch):
    devices = tuple(
        SimpleNamespace(ep_name="WebGpuExecutionProvider") for _ in range(8)
    )
    monkeypatch.setattr(execution_provider, "_webgpu_devices", lambda: devices)
    options = _SessionOptions()
    sessions = []
    session = object()

    def create_session(path, **kwargs):
        sessions.append((path, kwargs))
        if len(sessions) == 1:
            raise execution_provider.EPFail(
                "EP_FAIL: GetFusedActivationAttr(info, activation_).IsOK() was false."
            )
        return session

    monkeypatch.setattr(execution_provider.ort, "InferenceSession", create_session)

    with pytest.warns(RuntimeWarning, match="ConvActivationFusion disabled"):
        result = execution_provider.create_inference_session(
            "ocr.onnx", session_options=options
        )

    assert result is session
    assert sessions == [
        ("ocr.onnx", {"sess_options": options}),
        (
            "ocr.onnx",
            {
                "sess_options": options,
                "disabled_optimizers": ["ConvActivationFusion"],
            },
        ),
    ]
    assert options.providers == [(devices[:1], execution_provider._WEBGPU_OPTIONS)]


@pytest.mark.parametrize("providers", [None, ["CPUExecutionProvider"]])
def test_unrelated_session_failures_are_not_retried(monkeypatch, providers):
    device = SimpleNamespace(ep_name="WebGpuExecutionProvider")
    monkeypatch.setattr(execution_provider, "_webgpu_devices", lambda: (device,))
    monkeypatch.setattr(execution_provider.ort, "SessionOptions", _SessionOptions)
    sessions = []

    def create_session(*args, **kwargs):
        sessions.append((args, kwargs))
        raise execution_provider.EPFail("device initialization failed")

    monkeypatch.setattr(execution_provider.ort, "InferenceSession", create_session)

    with pytest.raises(execution_provider.EPFail, match="device initialization failed"):
        execution_provider.create_inference_session("model.onnx", providers=providers)

    assert len(sessions) == 1


def test_webgpu_retry_failure_is_propagated(monkeypatch):
    device = SimpleNamespace(ep_name="WebGpuExecutionProvider")
    monkeypatch.setattr(execution_provider, "_webgpu_devices", lambda: (device,))
    monkeypatch.setattr(execution_provider.ort, "SessionOptions", _SessionOptions)
    sessions = []

    def create_session(*args, **kwargs):
        sessions.append((args, kwargs))
        raise execution_provider.EPFail("GetFusedActivationAttr failed")

    monkeypatch.setattr(execution_provider.ort, "InferenceSession", create_session)

    with pytest.warns(RuntimeWarning, match="ConvActivationFusion disabled"):
        with pytest.raises(execution_provider.EPFail, match="GetFusedActivationAttr"):
            execution_provider.create_inference_session("model.onnx")

    assert len(sessions) == 2
