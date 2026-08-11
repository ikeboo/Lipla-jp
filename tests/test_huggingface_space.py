import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from lipla import LPDetResult

_MODULE_PATH = Path(__file__).resolve().parents[1] / "hf_space" / "space_inference.py"
_SPEC = spec_from_file_location("lipla_space_inference", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
recognize_image = _MODULE.recognize_image
result_to_dict = _MODULE.result_to_dict


def _result():
    result = LPDetResult(
        vertices=np.array([[1, 2], [1, 8], [18, 8], [18, 2]], dtype=np.float32),
        score=0.95,
        plate_image=np.full((5, 10, 3), 16, dtype=np.uint8),
        original_image=np.full((10, 20, 3), 32, dtype=np.uint8),
        class_name="private",
        area="品川",
        area_score=0.91,
        class_number="500",
        class_number_score=0.92,
        kana="あ",
        kana_score=0.93,
        number=1234,
        number_score=0.94,
    )
    result._det_image = np.array([[[10, 20, 30]]], dtype=np.uint8)
    result._result_image = np.array([[[40, 50, 60]]], dtype=np.uint8)
    return result


def test_result_to_dict_excludes_images_and_serializes_vertices():
    data = result_to_dict(_result())

    assert data == {
        "vertices": [[1.0, 2.0], [1.0, 8.0], [18.0, 8.0], [18.0, 2.0]],
        "score": 0.95,
        "class_name": "private",
        "area": "品川",
        "area_score": 0.91,
        "class_number": "500",
        "class_number_score": 0.92,
        "kana": "あ",
        "kana_score": 0.93,
        "number": 1234,
        "number_score": 0.94,
    }


def test_recognize_image_converts_color_and_returns_galleries_and_json():
    result = _result()
    received = []

    def recognizer(image):
        received.append(image)
        return [result]

    rgb_image = np.array([[[1, 2, 3]]], dtype=np.uint8)
    det_images, result_images, result_json = recognize_image(
        rgb_image, recognizer_factory=lambda: recognizer
    )

    assert np.array_equal(received[0], np.array([[[3, 2, 1]]], dtype=np.uint8))
    assert det_images[0][1] == "LPDetResult[0]"
    assert np.array_equal(det_images[0][0], np.array([[[30, 20, 10]]]))
    assert result_images[0][1] == "LPDetResult[0]"
    assert np.array_equal(result_images[0][0], np.array([[[60, 50, 40]]]))
    assert json.loads(result_json) == [result_to_dict(result)]
    assert "品川" in result_json


def test_recognize_image_clears_outputs_when_input_is_empty():
    assert recognize_image(None) == ([], [], "[]")
