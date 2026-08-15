import numpy as np
import pytest
from PIL import ImageFont

from lipla.core.recognition_result import LPDetResult, render_result_image


def _result():
    return LPDetResult(
        vertices=np.array([[1, 1], [1, 8], [18, 8], [18, 1]], dtype=np.float32),
        score=0.9,
        plate_image=np.full((20, 50, 3), 64, dtype=np.uint8),
        original_image=np.zeros((10, 20, 3), dtype=np.uint8),
        class_name="private",
        area="品川",
        area_score=0.8,
        class_number="500",
        class_number_score=0.8,
        kana="あ",
        kana_score=0.8,
        number=1234,
        number_score=0.8,
    )


def test_render_result_image_accepts_injected_font_loader():
    result = _result()
    sizes = []

    rendered = render_result_image(
        result,
        font_loader=lambda size: sizes.append(size) or ImageFont.load_default(),
    )

    assert sizes == [16]
    assert rendered.dtype == np.uint8
    assert rendered.shape[0] > result.plate_image.shape[0]
    assert np.array_equal(rendered[:20, :50], result.plate_image)


def test_detection_image_is_cached_without_modifying_original():
    result = _result()

    rendered = result.det_image

    assert result.det_image is rendered
    assert np.array_equal(result.original_image, np.zeros((10, 20, 3)))
    assert np.array_equal(rendered[5, 1], [255, 0, 255])


@pytest.mark.parametrize(
    ("image_width", "expected_thickness"),
    [(1000, 5), (20, 1)],
)
def test_detection_line_thickness_is_half_percent_of_original_width(
    monkeypatch, image_width, expected_thickness
):
    result = _result()
    result.original_image = np.zeros((10, image_width, 3), dtype=np.uint8)
    thicknesses = []

    def record_polylines(image, points, isClosed, color, thickness):
        thicknesses.append(thickness)
        return image

    monkeypatch.setattr("lipla.core.recognition_result.cv2.polylines", record_polylines)

    result.det_image

    assert thicknesses == [expected_thickness]
