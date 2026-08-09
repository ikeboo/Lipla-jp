import tomllib

import numpy as np
import pytest

from lipla.core.license_plate_recognizer import Recognizer
from lipla.core.plate_normalizer import PlateNormalizer
from lipla.inferencers.ec_pose import ECPose, PoseResult
from lipla.inferencers.ppocr import (
    CTCDecoder,
    OCRResult,
    crop_and_get_perspective,
    postprocess_dbnet,
    preprocess_det,
    preprocess_rec,
)


def test_runtime_dependencies_do_not_include_development_tools():
    with open("pyproject.toml", "rb") as file:
        project = tomllib.load(file)

    runtime_dependencies = {
        requirement.split(">", 1)[0].split("<", 1)[0]
        for requirement in project["project"]["dependencies"]
    }

    assert "pytest" not in runtime_dependencies
    assert "ipykernel" not in runtime_dependencies
    assert {"pytest", "ipykernel"}.issubset(project["dependency-groups"]["dev"])


def test_preprocess_det_returns_actual_axis_scales_after_rounding():
    image = np.zeros((100, 333, 3), dtype=np.uint8)

    tensor, scale = preprocess_det(image, target_size=736)

    assert tensor.shape == (1, 3, 224, 736)
    assert scale == pytest.approx((736 / 333, 224 / 100))


def test_postprocess_dbnet_restores_each_axis_with_its_own_scale(monkeypatch):
    from lipla.inferencers import ppocr

    box = np.array(
        [[10, 10], [40, 10], [40, 30], [10, 30]], dtype=np.float32
    )
    monkeypatch.setattr(
        ppocr.cv2,
        "findContours",
        lambda *_args, **_kwargs: ([np.ones((4, 1, 2), dtype=np.int32)], None),
    )
    monkeypatch.setattr(ppocr, "get_mini_boxes", lambda _contour: (box.copy(), 20))
    monkeypatch.setattr(ppocr, "box_score_fast", lambda *_args: 1.0)
    monkeypatch.setattr(ppocr, "unclip", lambda value, **_kwargs: value)

    boxes, scores = postprocess_dbnet(
        [np.ones((1, 1, 64, 64), dtype=np.float32)],
        original_shape=(20, 30),
        scale=(2.0, 4.0),
    )

    assert boxes == [[[5, 2], [20, 2], [20, 7], [5, 7]]]
    assert scores == [1.0]


@pytest.mark.parametrize(
    "function,image",
    [
        (preprocess_det, np.zeros((2, 2, 3), dtype=np.float32)),
        (preprocess_rec, np.zeros((2, 2, 3), dtype=np.float32)),
        (Recognizer._validate_image, np.zeros((2, 2, 3), dtype=np.float32)),
    ],
)
def test_model_inputs_require_uint8(function, image):
    with pytest.raises(TypeError, match="uint8"):
        function(image)


def test_preprocess_rec_keeps_very_thin_crop_nonempty():
    crop = np.zeros((100, 1, 3), dtype=np.uint8)

    tensor = preprocess_rec(crop, target_height=48)

    assert tensor.shape == (1, 3, 48, 1)


def test_perspective_crop_rejects_degenerate_points():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    points = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)

    with pytest.raises(ValueError, match="non-degenerate"):
        crop_and_get_perspective(image, points)


def test_result_containers_reject_mismatched_parallel_fields():
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="same length"):
        OCRResult(image=image, boxes=[], scores=[0.5], texts=[])
    with pytest.raises(ValueError, match="same length"):
        PoseResult(classes=[0], class_names=[], kpts=[], scores=[])


def test_ocr_result_uses_identity_equality_and_compact_repr():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    left = OCRResult(image=image, boxes=[], scores=[], texts=[])
    right = OCRResult(image=image.copy(), boxes=[], scores=[], texts=[])

    assert left != right
    assert "image=" not in repr(left)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_ecpose_rejects_invalid_threshold_before_loading_model(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        ECPose("unused.onnx", thresh=threshold)


def test_ctc_decoder_rejects_invalid_output_shape():
    decoder = object.__new__(CTCDecoder)
    decoder.character = ["<blank>", "A"]
    decoder.filter_masks = {
        name: np.array([False, True]) for name in decoder.FILTER_NAMES
    }

    with pytest.raises(ValueError, match="character count"):
        decoder.decode(np.zeros((1, 2, 3), dtype=np.float32))


def test_ctc_decoder_adds_new_area_characters_to_filter(tmp_path):
    dictionary_path = tmp_path / "dict.yml"
    dictionary_path.write_text(
        "PostProcess:\n  character_dict:\n    - 旧\n    - 新\n", encoding="utf-8"
    )
    characters_path = tmp_path / "characters.yml"
    characters_path.write_text(
        "areas: [旧]\nhiragana: []\nalphabet: []\nnumbers: []\nsymbols: []\n",
        encoding="utf-8",
    )

    decoder = CTCDecoder(dictionary_path, characters_path, new_area_names=["新"])

    assert decoder.filter_masks["areas"].tolist() == [False, True, True, False]


def test_plate_normalizer_rejects_invalid_size_and_vertices():
    with pytest.raises(ValueError, match="at least 2"):
        PlateNormalizer(width=0)

    normalizer = PlateNormalizer()
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    vertices = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-degenerate"):
        normalizer.normalize(image, vertices)


def test_pose_nms_ignores_nonfinite_scores_and_degenerate_polygons():
    valid = ([0, 0, 0, 10, 20, 10, 20, 0], 0.8, "private")
    detections = [
        ([0, 0, 0, 10, 20, 10, 20, 0], float("nan"), "private"),
        ([1, 1, 2, 2, 3, 3, 4, 4], 0.9, "private"),
        valid,
    ]

    assert Recognizer._pose_nms(detections) == [valid]
