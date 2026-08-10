from types import SimpleNamespace

import numpy as np

from lipla.core.fixed_field_recognizer import (
    FixedFieldRecognizer,
    ctc_log_probability,
)
from lipla.core.ocr_result_parser import OCRResultParser


class _Decoder:
    character = ["<blank>", "品", "あ", "い", "3", "0", "A"]

    def decode(self, _predictions, filters=None):
        if filters is None:
            return "品30A", 0.75
        if "symbols" in filters:
            return "12-34", 0.85
        return "30A", 0.8


class _Session:
    def __init__(self):
        probabilities = np.full((1, 5, 7), 0.01, dtype=np.float32)
        probabilities[0, :, 0] = 0.1
        probabilities[0, 1, 1] = 0.9
        probabilities[0, 2, 2] = 0.8
        probabilities[0, 2, 3] = 0.2
        self.output = probabilities

    def run(self, _output_names, _input_feed):
        return [self.output]


def _ocr_model(**overrides):
    values = {
        "decoder": _Decoder(),
        "rec_session": _Session(),
        "rec_input_name": "image",
        "rec_target_height": 48,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fixed_field_recognizer_returns_field_specific_candidates():
    parser = OCRResultParser(["品"], ["あ", "い"])
    recognizer = FixedFieldRecognizer(_ocr_model(), parser)

    candidates = recognizer.recognize(np.zeros((160, 320, 3), dtype=np.uint8))

    assert candidates["area"][0].text == "品"
    assert candidates["class_number"][0].text == "30A"
    assert candidates["number"][0].text == "12-34"
    assert candidates["kana"][0].text == "あ"
    assert any(value.text == "30A" for value in candidates["class_number"])


def test_fixed_field_recognizer_can_skip_unavailable_recognition_session():
    parser = OCRResultParser(["品"], ["あ"])
    model = SimpleNamespace(decoder=_Decoder())
    recognizer = FixedFieldRecognizer(model, parser)

    candidates = recognizer.recognize(np.zeros((160, 320, 3), dtype=np.uint8))

    assert all(not values for values in candidates.values())


def test_ctc_log_probability_prefers_high_probability_token():
    probabilities = np.array([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]], dtype=np.float64)
    log_probabilities = np.log(probabilities)

    first = ctc_log_probability(log_probabilities, (1,))
    second = ctc_log_probability(log_probabilities, (2,))

    assert first > second
