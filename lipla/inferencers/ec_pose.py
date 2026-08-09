"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
"""

import json
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .model_loader import (
    ECPOSE_MODEL_FILENAME,
    MODEL_REVISION,
    download_model_file,
)


@dataclass(slots=True)
class PoseResult:
    """1枚の画像から検出した複数オブジェクトの姿勢推定結果。"""

    classes: list[int]
    class_names: list[str]
    kpts: list[list[float]]
    scores: list[float]

    def __post_init__(self) -> None:
        lengths = {
            len(self.classes),
            len(self.class_names),
            len(self.kpts),
            len(self.scores),
        }
        if len(lengths) != 1:
            raise ValueError("pose result fields must have the same length")

    def __len__(self) -> int:
        return len(self.classes)


def _decode_keypoints(
    keypoints: np.ndarray, num_keypoints: int | None = None
) -> np.ndarray:
    """Decode keypoints into an ``(N, 2)`` xy array.

    Supports: [K,2], [K,3], [2K], [3K].
    """
    kpts = np.asarray(keypoints)

    if kpts.ndim == 2:
        if kpts.shape[1] < 2:
            return np.zeros((0, 2), dtype=np.float32)
        return kpts[:, :2].astype(np.float32)

    if kpts.ndim == 1:
        if num_keypoints and kpts.size == num_keypoints * 3:
            return kpts.reshape(num_keypoints, 3)[:, :2].astype(np.float32)
        if num_keypoints and kpts.size == num_keypoints * 2:
            return kpts.reshape(num_keypoints, 2).astype(np.float32)
        if kpts.size % 3 == 0:
            return kpts.reshape(-1, 3)[:, :2].astype(np.float32)
        if kpts.size % 2 == 0:
            return kpts.reshape(-1, 2).astype(np.float32)

    return np.zeros((0, 2), dtype=np.float32)


class ECPose:
    def __init__(
        self,
        onnx_path: str | Path | None = None,
        thresh: float = 0.4,
        providers: list[str] | None = None,
        *,
        cache_dir: str | Path | None = None,
        revision: str = MODEL_REVISION,
        local_files_only: bool = False,
    ):
        if not isinstance(thresh, Real) or isinstance(thresh, bool):
            raise TypeError("thresh must be a number")
        if not np.isfinite(thresh) or not 0.0 <= thresh <= 1.0:
            raise ValueError("thresh must be between 0 and 1")

        if onnx_path is None:
            onnx_path = download_model_file(
                ECPOSE_MODEL_FILENAME,
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=local_files_only,
            )
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_kwargs = {} if providers is None else {"providers": providers}
        self.session = ort.InferenceSession(
            str(onnx_path), session_options, **session_kwargs
        )
        self.thresh = float(thresh)
        metadata = self._model_metadata()
        self.input_size = self._get_input_size(metadata)
        self.num_keypoints = self._get_num_keypoints(metadata)
        self.remap_categories = bool(metadata.get("remap_categories", False))

        category_ids = metadata.get("category_ids", [])
        category_names = metadata.get("category_names", [])
        if category_names and len(category_ids) != len(category_names):
            raise RuntimeError(
                "category_ids and category_names metadata must have the same length"
            )
        self.category_ids = [int(value) for value in category_ids]
        if not category_names:
            category_names = [str(value) for value in self.category_ids]
        self.class_names = {
            category_id: str(name)
            for category_id, name in zip(self.category_ids, category_names)
        }

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        model_inputs = self.session.get_inputs()
        if not model_inputs:
            raise RuntimeError("the ONNX model has no inputs")
        self.image_input_name = model_inputs[0].name
        self.size_input_name = model_inputs[1].name if len(model_inputs) > 1 else None
        self.output_names = [o.name for o in self.session.get_outputs()]

    def __call__(self, image: np.ndarray) -> PoseResult:
        tensor, target_size = self._preprocess(image)
        input_feed = {self.image_input_name: tensor}
        if self.size_input_name is not None:
            input_feed[self.size_input_name] = target_size
        outputs = self.session.run(output_names=None, input_feed=input_feed)
        return self._postprocess(outputs, image.shape[:2])

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a BGR image with shape (H, W, 3)")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("image must not be empty")
        if image.dtype != np.uint8:
            raise TypeError("image must have dtype uint8")

        height, width = image.shape[:2]
        input_height, input_width = self.input_size
        resized = cv2.resize(
            image,
            (input_width, input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
        target_size = np.array([[width, height]], dtype=np.int64)
        return np.ascontiguousarray(tensor), target_size

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        original_shape: tuple[int, int],
    ) -> PoseResult:
        original_height, original_width = original_shape
        scores, labels, keypoints = self._parse_outputs(outputs)
        scores = scores[0]
        labels = labels[0]
        keypoints = keypoints[0]

        classes = []
        class_names = []
        result_kpts = []
        result_scores = []
        for index in np.flatnonzero(scores > self.thresh):
            mapped_label = self._map_label(int(labels[index]))
            if mapped_label is None:
                continue
            label, class_name = mapped_label
            xy = _decode_keypoints(keypoints[index], num_keypoints=self.num_keypoints)
            if len(xy) != self.num_keypoints:
                raise RuntimeError(
                    f"Expected {self.num_keypoints} keypoints, got {len(xy)} "
                    f"from shape {keypoints[index].shape}"
                )

            xy[:, 0] = np.clip(xy[:, 0], 0, original_width - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, original_height - 1)

            classes.append(label)
            class_names.append(class_name or str(label))
            result_kpts.append(xy.reshape(-1).astype(float).tolist())
            result_scores.append(float(scores[index]))

        return PoseResult(
            classes=classes,
            class_names=class_names,
            kpts=result_kpts,
            scores=result_scores,
        )

    def _get_input_size(self, metadata: dict) -> tuple[int, int]:
        input_shape = self.session.get_inputs()[0].shape
        if (
            len(input_shape) == 4
            and isinstance(input_shape[2], int)
            and isinstance(input_shape[3], int)
            and input_shape[2] > 0
            and input_shape[3] > 0
        ):
            return int(input_shape[2]), int(input_shape[3])

        metadata_size = metadata.get("input_size")
        if (
            isinstance(metadata_size, list)
            and len(metadata_size) == 2
            and all(isinstance(value, int) for value in metadata_size)
            and all(value > 0 for value in metadata_size)
        ):
            return int(metadata_size[0]), int(metadata_size[1])
        raise RuntimeError("Could not determine input size from the ONNX model")

    def _get_num_keypoints(self, metadata: dict) -> int:
        for model_output in self.session.get_outputs():
            if model_output.name == "keypoints":
                output_shape = model_output.shape
                if len(output_shape) >= 2 and isinstance(output_shape[-2], int):
                    count = int(output_shape[-2])
                    if count > 0:
                        return count

        metadata_num_keypoints = metadata.get("num_keypoints")
        if isinstance(metadata_num_keypoints, int) and metadata_num_keypoints > 0:
            return metadata_num_keypoints
        raise RuntimeError("Could not determine keypoint count from the ONNX model")

    def _model_metadata(self):
        raw_metadata = self.session.get_modelmeta().custom_metadata_map
        metadata = {}
        for key, value in raw_metadata.items():
            try:
                metadata[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                metadata[key] = value
        return metadata

    def _map_label(self, raw_label: int):
        if not self.category_ids:
            return raw_label, None
        if self.remap_categories:
            if not 0 <= raw_label < len(self.category_ids):
                return None
            category_id = self.category_ids[raw_label]
        else:
            if raw_label not in self.class_names:
                return None
            category_id = raw_label
        return category_id, self.class_names.get(category_id)

    def _parse_outputs(self, outputs):
        mapping = {name: arr for name, arr in zip(self.output_names, outputs)}

        if {"scores", "labels", "keypoints"}.issubset(mapping.keys()):
            scores = mapping["scores"]
            labels = mapping["labels"]
            keypoints = mapping["keypoints"]
        elif len(outputs) == 3:
            scores, labels, keypoints = outputs
        else:
            raise RuntimeError(
                f"Unexpected ONNX outputs. names={self.output_names}, count={len(outputs)}; expected scores/labels/keypoints"
            )

        scores = np.asarray(scores)
        labels = np.asarray(labels)
        keypoints = np.asarray(keypoints)
        if scores.ndim != 2 or labels.ndim != 2 or keypoints.ndim < 3:
            raise RuntimeError("ONNX pose outputs have invalid dimensions")
        if scores.shape[0] != 1 or labels.shape[0] != 1 or keypoints.shape[0] != 1:
            raise RuntimeError("ECPose only supports a batch size of one")
        if not (scores.shape[1] == labels.shape[1] == keypoints.shape[1]):
            raise RuntimeError("ONNX pose outputs have inconsistent detection counts")
        return scores, labels, keypoints
