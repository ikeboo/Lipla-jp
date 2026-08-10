import numpy as np
import pytest

from lipla.core.ocr_result_parser import (
    OCRResultParser,
    TextCandidate,
    edit_distance,
)
from lipla.inferencers.ppocr import OCRResult


def _box(center_x, center_y):
    return [
        [center_x - 10, center_y - 5],
        [center_x + 10, center_y - 5],
        [center_x + 10, center_y + 5],
        [center_x - 10, center_y + 5],
    ]


def test_parser_splits_spatial_rows_into_plate_fields():
    parser = OCRResultParser(["品川", "横浜"], ["あ", "い"])
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    ocr_result = OCRResult(
        image=image,
        boxes=[_box(80, 25), _box(100, 75)],
        scores=[0.9, 0.8],
        texts=["品川30A", "あ 12-34"],
    )

    fields = parser.parse(ocr_result)

    assert fields.area == TextCandidate("品川", 0.9)
    assert fields.class_number == TextCandidate("30A", 0.9)
    assert fields.kana == TextCandidate("あ", 0.8)
    assert fields.number == TextCandidate("1234", 0.8)


def test_parser_combines_selected_spatial_and_fixed_candidates():
    parser = OCRResultParser(["品川"], ["あ"])
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    ocr_result = OCRResult(
        image=image,
        boxes=[_box(100, 75)],
        scores=[0.95],
        texts=["あ 9999"],
    )

    fields = parser.parse(
        ocr_result,
        field_candidates={
            "class_number": [TextCandidate("500", 0.8)],
            "number": [TextCandidate("12-34", 0.7)],
        },
        spatial_fields=["kana"],
    )

    assert fields.class_number.text == "500"
    assert fields.kana.text == "あ"
    assert fields.number.text == "1234"


def test_parser_corrects_similar_area_but_rejects_unrelated_noise():
    parser = OCRResultParser(["品川", "横浜"], ["あ"])

    corrected = parser.best_area([TextCandidate("品河", 0.9)])
    rejected = parser.best_area([TextCandidate("東京", 0.9)])

    assert corrected.text == "品川"
    assert corrected.score == pytest.approx(0.45)
    assert rejected == TextCandidate("", 0.0)


def test_parser_rejects_unknown_candidate_field():
    parser = OCRResultParser(["品川"], ["あ"])
    ocr_result = OCRResult(
        image=np.zeros((20, 40, 3), dtype=np.uint8),
        boxes=[],
        scores=[],
        texts=[],
    )

    with pytest.raises(ValueError, match="Unknown candidate fields"):
        parser.parse(ocr_result, field_candidates={"unknown": []})


@pytest.mark.parametrize(
    ("left", "right", "distance"),
    [("品川", "品川", 0), ("品河", "品川", 1), ("横浜", "横浜新", 1)],
)
def test_edit_distance(left, right, distance):
    assert edit_distance(left, right) == distance
