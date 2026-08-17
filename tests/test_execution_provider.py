import sys
from types import SimpleNamespace

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
