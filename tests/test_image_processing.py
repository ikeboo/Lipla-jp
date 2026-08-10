import cv2
import numpy as np
import pytest

from lipla.core.image_processing import (
    as_bgr,
    pad_for_ocr,
    preprocess_local_plate,
    read_bgr_image,
    remove_padding,
    validate_bgr_image,
)


def test_read_bgr_image_supports_japanese_path(tmp_path):
    source = np.full((6, 10, 3), (10, 20, 30), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", source)
    assert success
    path = tmp_path / "日本語の画像.png"
    path.write_bytes(encoded.tobytes())

    loaded = read_bgr_image(path)

    assert np.array_equal(loaded, source)


def test_as_bgr_converts_grayscale_and_reuses_bgr_image():
    grayscale = np.arange(12, dtype=np.uint8).reshape(3, 4)
    bgr = as_bgr(grayscale)

    assert bgr.shape == (3, 4, 3)
    assert np.array_equal(bgr[..., 0], grayscale)
    assert as_bgr(bgr) is bgr


def test_preprocess_local_plate_keeps_dark_text_and_whitens_artwork():
    hsv = np.array([[[0, 0, 20], [60, 100, 100], [120, 255, 220]]], dtype=np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    filtered = preprocess_local_plate(image)

    assert np.array_equal(filtered[0, 0], image[0, 0])
    assert np.array_equal(filtered[0, 1], image[0, 1])
    assert np.array_equal(filtered[0, 2], [255, 255, 255])


def test_ocr_padding_can_be_removed_without_changing_plate():
    image = np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3)

    padded, padding = pad_for_ocr(image)

    assert padding == 8
    assert np.array_equal(remove_padding(padded, padding), image)


@pytest.mark.parametrize(
    "image,error",
    [
        ([], TypeError),
        (np.zeros((2, 2), dtype=np.uint8), ValueError),
        (np.zeros((2, 2, 3), dtype=np.float32), TypeError),
    ],
)
def test_validate_bgr_image_rejects_invalid_inputs(image, error):
    with pytest.raises(error):
        validate_bgr_image(image)
