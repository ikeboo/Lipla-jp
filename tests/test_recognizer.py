from types import SimpleNamespace

import numpy as np
import yaml


class _PoseModel:
    calls = []
    images = []

    def __init__(self, model_path, **kwargs):
        self.calls.append((model_path, kwargs))

    def __call__(self, image):
        self.images.append(image)
        return SimpleNamespace(kpts=[], scores=[], class_names=[])


class _OCRModel:
    calls = []

    def __init__(
        self, det_model_path, rec_model_path, *, characters_path, **kwargs
    ):
        self.calls.append(
            (det_model_path, rec_model_path, characters_path, kwargs)
        )
        with characters_path.open("r", encoding="utf-8") as file:
            characters = yaml.safe_load(file)
        area_characters = dict.fromkeys(
            character
            for area in characters["areas"]
            for character in str(area)
        )
        self.decoder = SimpleNamespace(
            character=["<blank>", *area_characters]
        )

    def __call__(self, _image):
        raise AssertionError("OCR must not run when no plate is detected")


def test_readme_recognizer_api_uses_default_downloads(monkeypatch):
    import lipla
    from lipla.core import license_plate_recognizer
    from lipla.inferencers.model_loader import MODEL_REVISION

    _PoseModel.calls.clear()
    _PoseModel.images.clear()
    _OCRModel.calls.clear()
    monkeypatch.setattr(license_plate_recognizer, "ECPose", _PoseModel)
    monkeypatch.setattr(license_plate_recognizer, "PPOCR", _OCRModel)

    rec = lipla.Recognizer()
    results = rec(np.zeros((32, 64, 3), dtype=np.uint8))

    assert results == []
    shared_options = {
        "providers": None,
        "cache_dir": None,
        "revision": MODEL_REVISION,
        "local_files_only": False,
    }
    assert _PoseModel.calls == [(None, shared_options)]
    assert len(_OCRModel.calls) == 1
    det_path, rec_path, characters_path, ocr_options = _OCRModel.calls[0]
    assert det_path is None
    assert rec_path is None
    assert characters_path.name == "characters.yml"
    assert ocr_options == {
        "dict_path": None,
        "new_area_names": None,
        **shared_options,
    }


def test_recognizer_adds_new_area_names_to_ocr_and_vocabulary(monkeypatch):
    import lipla
    from lipla.core import license_plate_recognizer

    _PoseModel.calls.clear()
    _OCRModel.calls.clear()
    monkeypatch.setattr(license_plate_recognizer, "ECPose", _PoseModel)
    monkeypatch.setattr(license_plate_recognizer, "PPOCR", _OCRModel)

    rec = lipla.Recognizer(new_area_names=["札幌新", "札幌"])

    assert rec.areas[-1] == "札幌新"
    assert rec.areas.count("札幌") == 1
    assert _OCRModel.calls[0][3]["new_area_names"] == ["札幌新", "札幌"]


def test_recognizer_accepts_path_containing_japanese_characters(
    monkeypatch, tmp_path
):
    import cv2

    import lipla
    from lipla.core import license_plate_recognizer

    _PoseModel.calls.clear()
    _PoseModel.images.clear()
    _OCRModel.calls.clear()
    monkeypatch.setattr(license_plate_recognizer, "ECPose", _PoseModel)
    monkeypatch.setattr(license_plate_recognizer, "PPOCR", _OCRModel)

    source_image = np.zeros((12, 24, 3), dtype=np.uint8)
    source_image[:, :8] = (10, 20, 30)
    success, encoded = cv2.imencode(".png", source_image)
    assert success
    image_path = tmp_path / "日本語を含む画像.png"
    image_path.write_bytes(encoded.tobytes())

    rec = lipla.Recognizer()
    results = rec(str(image_path))

    assert results == []
    assert len(_PoseModel.images) == 1
    assert np.array_equal(_PoseModel.images[0], source_image)


def test_system_japanese_font_falls_back_to_next_candidate(monkeypatch):
    from lipla.core import license_plate_recognizer

    loaded_font = object()
    calls = []

    def load_font(font_name, font_size):
        calls.append((font_name, font_size))
        if font_name == "unavailable.ttf":
            raise OSError("font not found")
        return loaded_font

    monkeypatch.setattr(
        license_plate_recognizer,
        "_SYSTEM_JAPANESE_FONT_CANDIDATES",
        ("unavailable.ttf", "system-japanese.ttf"),
    )
    monkeypatch.setattr(
        license_plate_recognizer.ImageFont, "truetype", load_font
    )

    assert license_plate_recognizer._load_system_japanese_font(24) is loaded_font
    assert calls == [("unavailable.ttf", 24), ("system-japanese.ttf", 24)]


def test_system_japanese_font_error_lists_attempted_fonts(monkeypatch):
    from lipla.core import license_plate_recognizer

    monkeypatch.setattr(
        license_plate_recognizer,
        "_SYSTEM_JAPANESE_FONT_CANDIDATES",
        ("first.ttf", "second.ttf"),
    )

    def font_not_found(_font_name, _font_size):
        raise OSError("font not found")

    monkeypatch.setattr(
        license_plate_recognizer.ImageFont, "truetype", font_not_found
    )

    try:
        license_plate_recognizer._load_system_japanese_font(24)
    except OSError as error:
        assert "first.ttf, second.ttf" in str(error)
    else:
        raise AssertionError("missing system fonts must raise OSError")


def _make_detection_result():
    from lipla import LPDetResult

    return LPDetResult(
        vertices=np.array(
            [[5, 5], [5, 25], [45, 25], [45, 5]], dtype=np.float32
        ),
        score=0.95,
        plate_image=np.full((40, 100, 3), 64, dtype=np.uint8),
        original_image=np.zeros((30, 50, 3), dtype=np.uint8),
        class_name="private",
        area="Tokyo",
        area_score=0.9,
        class_number="123",
        class_number_score=0.9,
        kana="A",
        kana_score=0.9,
        number=4567,
        number_score=0.9,
    )


def test_det_image_is_created_lazily_and_does_not_modify_original():
    result = _make_detection_result()

    assert result._det_image is None

    det_image = result.det_image

    assert result.det_image is det_image
    assert np.array_equal(result.original_image, np.zeros((30, 50, 3)))
    assert np.array_equal(det_image[15, 5], [255, 0, 255])
    assert np.array_equal(det_image[15, 25], [0, 0, 0])


def test_result_image_is_created_lazily_and_used_by_compatibility_api(
    monkeypatch,
):
    from PIL import ImageFont

    from lipla.core import license_plate_recognizer

    result = _make_detection_result()
    font = ImageFont.load_default()
    monkeypatch.setattr(
        license_plate_recognizer,
        "_load_system_japanese_font",
        lambda _font_size: font,
    )

    assert result._result_image is None

    result_image = result.result_image
    recognizer = object.__new__(license_plate_recognizer.Recognizer)

    assert result.result_image is result_image
    assert recognizer.create_result_images([result]) == [result_image]
    assert np.array_equal(result_image[:40], result.plate_image)


def test_visualize_displays_detection_and_recognition_images(monkeypatch):
    import matplotlib.pyplot as plt

    class Axes:
        def __init__(self):
            self.image = None
            self.title = None
            self.axis_setting = None

        def imshow(self, image):
            self.image = image

        def set_title(self, title):
            self.title = title

        def axis(self, setting):
            self.axis_setting = setting

    result = _make_detection_result()
    det_image = np.full((3, 4, 3), (10, 20, 30), dtype=np.uint8)
    result_image = np.full((2, 5, 3), (40, 50, 60), dtype=np.uint8)
    result._det_image = det_image
    result._result_image = result_image
    ax1, ax2 = Axes(), Axes()
    subplot_calls = []
    show_calls = []

    def subplots(*args, **kwargs):
        subplot_calls.append((args, kwargs))
        return object(), (ax1, ax2)

    monkeypatch.setattr(plt, "subplots", subplots)
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    result.visualize()

    assert subplot_calls == [((1, 2), {"figsize": (10, 5)})]
    assert np.array_equal(ax1.image, det_image[..., ::-1])
    assert ax1.title == "Detected License Plate"
    assert ax1.axis_setting == "off"
    assert np.array_equal(ax2.image, result_image[..., ::-1])
    assert ax2.title == "Recognition Result"
    assert ax2.axis_setting == "off"
    assert show_calls == [True]
