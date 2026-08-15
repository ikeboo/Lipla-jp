"""姿勢推定によるナンバープレート検出結果の後処理。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import cv2
import numpy as np

PoseDetection: TypeAlias = tuple[list[float], float, str]

_MAX_OPPOSITE_EDGE_ANGLE_DEGREES = 20.0
_ANGLE_COMPARISON_TOLERANCE_DEGREES = 1e-5
_MIN_OPPOSITE_EDGE_LENGTH_RATIO = 0.25
_MAX_OPPOSITE_EDGE_LENGTH_RATIO = 1.2
_MIN_EDGE_LENGTH = 1e-6


def _angle_between_lines_degrees(left: np.ndarray, right: np.ndarray) -> float:
    """2つの線分が成す0度から90度までの角度を返す。"""
    cosine = abs(float(np.dot(left, right))) / (
        float(np.linalg.norm(left)) * float(np.linalg.norm(right))
    )
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def _has_plausible_plate_shape(polygon: np.ndarray) -> bool:
    """四角形の対辺の平行度と長さ比がプレートとして妥当か判定する。"""
    vertices = np.asarray(polygon, dtype=np.float64)
    edges = np.roll(vertices, -1, axis=0) - vertices
    edge_lengths = np.linalg.norm(edges, axis=1)
    if np.any(edge_lengths <= _MIN_EDGE_LENGTH):
        return False

    if _angle_between_lines_degrees(edges[0], edges[2]) >= (
        _MAX_OPPOSITE_EDGE_ANGLE_DEGREES - _ANGLE_COMPARISON_TOLERANCE_DEGREES
    ) or _angle_between_lines_degrees(edges[1], edges[3]) >= (
        _MAX_OPPOSITE_EDGE_ANGLE_DEGREES - _ANGLE_COMPARISON_TOLERANCE_DEGREES
    ):
        return False

    first_pair_mean = float((edge_lengths[0] + edge_lengths[2]) / 2.0)
    second_pair_mean = float((edge_lengths[1] + edge_lengths[3]) / 2.0)
    length_ratio = first_pair_mean / second_pair_mean
    return (
        _MIN_OPPOSITE_EDGE_LENGTH_RATIO < length_ratio < _MAX_OPPOSITE_EDGE_LENGTH_RATIO
    )


def polygon_iou(left: np.ndarray, right: np.ndarray) -> float:
    """2つの凸多角形のIoUを計算する。

    Args:
        left: 1つ目の多角形の頂点座標。
        right: 2つ目の多角形の頂点座標。

    Returns:
        2つの多角形のIoU。面積がない場合は ``0.0``。
    """
    left_hull = cv2.convexHull(np.asarray(left, dtype=np.float32))
    right_hull = cv2.convexHull(np.asarray(right, dtype=np.float32))
    left_area = abs(float(cv2.contourArea(left_hull)))
    right_area = abs(float(cv2.contourArea(right_hull)))
    if left_area <= 0.0 or right_area <= 0.0:
        return 0.0

    intersection_area, _ = cv2.intersectConvexConvex(left_hull, right_hull)
    union_area = left_area + right_area - float(intersection_area)
    return float(intersection_area) / union_area if union_area > 0.0 else 0.0


def suppress_duplicate_detections(
    detections: Sequence[PoseDetection],
    iou_threshold: float = 0.3,
) -> list[PoseDetection]:
    """重複する姿勢推定結果を信頼度順に除外する。

    不正な頂点、非有限値、面積を持たない四角形に加えて、対辺の角度や
    長さの比率がナンバープレートとして不自然な四角形も同時に除外する。

    Args:
        detections: 頂点座標、信頼度、クラス名からなる検出結果。
        iou_threshold: 同一物体とみなすIoUの下限。

    Returns:
        重複を除外した検出結果。信頼度の降順で返す。

    Raises:
        TypeError: ``iou_threshold`` が数値ではない場合。
        ValueError: ``iou_threshold`` が0から1の範囲外の場合。
    """
    if not isinstance(iou_threshold, (int, float)) or isinstance(iou_threshold, bool):
        raise TypeError("iou_threshold must be a number")
    if not np.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")

    valid_detections: list[tuple[PoseDetection, np.ndarray]] = []
    for raw_vertices, raw_score, class_name in detections:
        try:
            score = float(raw_score)
            polygon = np.asarray(raw_vertices, dtype=np.float32).reshape(4, 2)
        except (TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite(score) or not np.all(np.isfinite(polygon)):
            continue
        if abs(float(cv2.contourArea(polygon))) <= 1e-6:
            continue
        if not cv2.isContourConvex(polygon):
            continue
        if not _has_plausible_plate_shape(polygon):
            continue
        valid_detections.append(((raw_vertices, score, class_name), polygon))

    kept: list[PoseDetection] = []
    kept_polygons: list[np.ndarray] = []
    for detection, polygon in sorted(
        valid_detections, key=lambda item: item[0][1], reverse=True
    ):
        if all(polygon_iou(polygon, other) < iou_threshold for other in kept_polygons):
            kept.append(detection)
            kept_polygons.append(polygon)
    return kept
