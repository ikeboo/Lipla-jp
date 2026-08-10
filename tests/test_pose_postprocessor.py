import numpy as np
import pytest

from lipla.core.pose_postprocessor import (
    polygon_iou,
    suppress_duplicate_detections,
)


def _vertices(left, top, right, bottom):
    return [left, top, left, bottom, right, bottom, right, top]


def test_polygon_iou_returns_expected_overlap_ratio():
    left = np.array(_vertices(0, 0, 10, 10)).reshape(4, 2)
    right = np.array(_vertices(5, 0, 15, 10)).reshape(4, 2)

    assert polygon_iou(left, right) == pytest.approx(1 / 3)


def test_suppress_duplicate_detections_keeps_highest_score_and_order():
    duplicate_low = (_vertices(1, 0, 11, 10), 0.7, "private")
    independent = (_vertices(20, 0, 30, 10), 0.8, "local")
    duplicate_high = (_vertices(0, 0, 10, 10), 0.9, "private")

    kept = suppress_duplicate_detections([duplicate_low, independent, duplicate_high])

    assert kept == [duplicate_high, independent]


def test_suppress_duplicate_detections_ignores_invalid_polygons():
    valid = (_vertices(0, 0, 10, 10), 0.8, "private")
    detections = [
        (_vertices(0, 0, 10, 10), float("nan"), "private"),
        ([1, 1, 2, 2, 3, 3, 4, 4], 0.9, "private"),
        valid,
    ]

    assert suppress_duplicate_detections(detections) == [valid]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_suppress_duplicate_detections_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        suppress_duplicate_detections([], threshold)
