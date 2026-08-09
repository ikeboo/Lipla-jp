"""End-to-end recognition for Japanese license plates."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from lipla.inferencers.ec_pose import ECPose
from lipla.inferencers.model_loader import MODEL_REVISION
from lipla.inferencers.ppocr import PPOCR, OCRResult, preprocess_rec

from .plate_normalizer import PlateNormalizer

if sys.platform == "darwin":
    _SYSTEM_JAPANESE_FONT_CANDIDATES = (
        "Hiragino Sans GB.ttc",
        "ヒラギノ角ゴシック W3.ttc",
        "Arial Unicode.ttf",
    )
elif sys.platform == "win32":
    _SYSTEM_JAPANESE_FONT_CANDIDATES = (
        "YuGothR.ttc",
        "meiryo.ttc",
        "msgothic.ttc",
    )
else:
    _SYSTEM_JAPANESE_FONT_CANDIDATES = (
        "NotoSansCJK-Regular.ttc",
        "NotoSansCJKjp-Regular.otf",
        "NotoSansJP-Regular.ttf",
        "NotoSansJP-VariableFont_wght.ttf",
        "ipaexg.ttf",
        "ipag.ttf",
        "VL-Gothic-Regular.ttf",
        "TakaoPGothic.ttf",
    )


def _load_system_japanese_font(font_size: int) -> ImageFont.FreeTypeFont:
    """Load a Japanese-capable font installed by the operating system."""
    last_error: OSError | None = None
    for font_name in _SYSTEM_JAPANESE_FONT_CANDIDATES:
        try:
            # Pillow resolves bare filenames in the platform's font folders.
            return ImageFont.truetype(font_name, font_size)
        except OSError as error:
            last_error = error

    candidates = ", ".join(_SYSTEM_JAPANESE_FONT_CANDIDATES)
    raise OSError(
        "No Japanese-capable system font was found. "
        f"Install a Japanese font such as Noto Sans CJK. Tried: {candidates}"
    ) from last_error


@dataclass(slots=True, eq=False)
class LPDetResult:
    """A detected plate and its recognized fields.

    ``vertices`` keeps the pose model order: top-left, bottom-left,
    bottom-right, top-right.  ``number`` is an integer, so for example both
    ``12-34`` and ``・・ 12`` become ``1234`` and ``12`` respectively.  Zero
    means that the number could not be recognized (real plate numbers start
    at one).
    """

    vertices: np.ndarray
    score: float
    plate_image: np.ndarray = field(repr=False)
    original_image: np.ndarray = field(repr=False)
    class_name: str
    area: str
    area_score: float
    class_number: str
    class_number_score: float
    kana: str
    kana_score: float
    number: int
    number_score: float
    _det_image: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _result_image: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float32)
        if vertices.shape != (4, 2):
            raise ValueError("vertices must have shape (4, 2)")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices must contain only finite values")
        _validate_bgr_image(self.plate_image)
        _validate_bgr_image(self.original_image)
        self.vertices = np.ascontiguousarray(vertices)

    @property
    def det_image(self) -> np.ndarray:
        """Original image with the detected plate outlined in magenta."""
        if self._det_image is None:
            _validate_bgr_image(self.original_image)
            det_image = self.original_image.copy()
            vertices = np.rint(self.vertices).astype(np.int32).reshape(4, 1, 2)
            cv2.polylines(
                det_image,
                [vertices],
                isClosed=True,
                color=(255, 0, 255),
                thickness=2,
            )
            self._det_image = det_image
        return self._det_image

    @property
    def result_image(self) -> np.ndarray:
        """Plate image with its recognized text rendered underneath."""
        if self._result_image is None:
            self._result_image = _create_result_image(self)
        return self._result_image

    def visualize(self) -> None:
        """Display the detection and recognition result side by side."""
        import matplotlib.pyplot as plt

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(self.det_image[..., ::-1])
        ax1.set_title("Detected License Plate")
        ax1.axis("off")
        ax2.imshow(self.result_image[..., ::-1])
        ax2.set_title("Recognition Result")
        ax2.axis("off")
        plt.show()


def _validate_bgr_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image with shape (H, W, 3)")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")
    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")


def _create_result_image(result: LPDetResult) -> np.ndarray:
    """Render a recognized plate and its text as a BGR image."""
    _validate_bgr_image(result.plate_image)
    plate = np.ascontiguousarray(result.plate_image)
    plate_height, plate_width = plate.shape[:2]
    font_size = max(16, int(round(plate_height * 0.20)))
    font = _load_system_japanese_font(font_size)
    text = f"{result.area} {result.class_number}\n{result.kana} {result.number}"

    # Measure the two lines first so that the lower area always has enough
    # room even if a non-standard plate image is supplied.
    measuring_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_box = measuring_draw.multiline_textbbox(
        (0, 0), text, font=font, spacing=4, align="center"
    )
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    padding = max(8, int(round(font_size * 0.4)))
    canvas_width = max(plate_width, text_width + padding * 2)
    canvas_height = plate_height + text_height + padding * 2

    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
    plate_rgb = Image.fromarray(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
    canvas.paste(plate_rgb, ((canvas_width - plate_width) // 2, 0))

    draw = ImageDraw.Draw(canvas)
    text_x = (canvas_width - text_width) / 2 - text_box[0]
    text_y = plate_height + padding - text_box[1]
    draw.multiline_text(
        (text_x, text_y),
        text,
        font=font,
        fill="black",
        spacing=4,
        align="center",
    )
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


@dataclass(slots=True)
class _TextCandidate:
    text: str
    score: float


class Recognizer:
    """Detect, rectify and read all Japanese license plates in a BGR image.

    PP-OCR's text detector is useful for finding a whole row, while its boxes
    are not stable enough to separate the large kana from the serial number.
    Japanese plates have a fixed two-row layout, therefore this recognizer
    combines whole-plate OCR with field-specific recognition of four fixed
    regions.  Character masks and a vocabulary of valid issuing areas are
    then used to reject impossible readings.
    """

    # Ratios in a 2:1 rectified plate.  They deliberately overlap a little so
    # small pose errors do not clip glyphs.
    _FIELD_RECTS = {
        "area": (0.194, 0.00, 0.519, 0.325),
        "class_number": (0.51, 0.00, 0.81, 0.36),
        "number": (0.15, 0.34, 1.00, 1.00),
    }
    # Kana is the field most affected by a few pixels of pose error.  The
    # first crop is tight; the other two provide lower-weight context above
    # and to the sides.  Values were selected on dataset/val_images.
    _KANA_RECTS = (
        ((0.009, 0.438, 0.197, 1.00), 1.0),
        ((0.016, 0.469, 0.197, 1.00), 0.7),
        ((0.000, 0.375, 0.219, 1.00), 0.7),
    )
    _FIELD_FILTERS = {
        "class_number": ["numbers", "alphabet"],
        "number": ["numbers", "symbols"],
    }
    _ALNUM_RE = re.compile(r"[^0-9A-Z]")
    _DIGIT_RE = re.compile(r"\d")
    _CLASS_SUFFIX_CHARACTERS = frozenset("0123456789ACFHKLMPXY")
    # OpenCV hue uses [0, 179].  These thresholds keep black and dark green
    # plate text while rejecting the colorful illustrations on local plates.
    # They were selected on all 611 local annotations in dataset/val_labels.json.
    _LOCAL_BLACK_VALUE_MAX = 50
    _LOCAL_GREEN_HUE_MIN = 30
    _LOCAL_GREEN_HUE_MAX = 95
    _LOCAL_GREEN_SATURATION_MIN = 25
    _LOCAL_GREEN_VALUE_MAX = 130
    _LOCAL_SPATIAL_FIELDS = frozenset(("area", "class_number", "kana"))

    def __init__(
        self,
        pose_model_path: str | Path | None = None,
        ocr_det_model_path: str | Path | None = None,
        ocr_rec_model_path: str | Path | None = None,
        *,
        ocr_dict_path: str | Path | None = None,
        new_area_names: list[str] | None = None,
        providers: list[str] | None = None,
        cache_dir: str | Path | None = None,
        revision: str = MODEL_REVISION,
        local_files_only: bool = False,
    ):
        characters_path = (
            Path(__file__).resolve().parents[1] / "configs" / "characters.yml"
        )
        self.pose_model = ECPose(
            pose_model_path,
            providers=providers,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
        )
        self.plate_normalizer = PlateNormalizer()
        self.ocr_model = PPOCR(
            ocr_det_model_path,
            ocr_rec_model_path,
            dict_path=ocr_dict_path,
            characters_path=characters_path,
            new_area_names=new_area_names,
            providers=providers,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
        )

        with characters_path.open("r", encoding="utf-8") as file:
            characters = yaml.safe_load(file)
        area_names = [*characters["areas"], *(new_area_names or ())]
        self.areas = tuple(dict.fromkeys(map(str, area_names)))
        self.kana_characters = tuple(
            dict.fromkeys(str(value) for value in characters["hiragana"])
        )
        self.hiragana = frozenset(self.kana_characters)
        decoder_characters = self.ocr_model.decoder.character
        self._area_tokens = tuple(
            (
                area,
                tuple(decoder_characters.index(character) for character in area),
            )
            for area in self.areas
        )

    def __call__(self, image: np.ndarray | str | Path) -> list[LPDetResult]:
        """Recognize every plate in ``image``.

        Args:
            image: OpenCV BGR image with shape ``(height, width, 3)``, or a
                path to an image file. Paths containing Japanese characters
                are supported.

        Returns:
            Results in the detector's confidence order.  An empty list is
            returned when no plate is detected.
        """
        if isinstance(image, (str, Path)):
            image = self._read_image(image)
        self._validate_image(image)
        pose_result = self.pose_model(image)
        results: list[LPDetResult] = []

        lengths = {
            len(pose_result.kpts),
            len(pose_result.scores),
            len(pose_result.class_names),
        }
        if len(lengths) != 1:
            raise RuntimeError("pose result fields must have the same length")

        detections = list(
            zip(
                pose_result.kpts,
                pose_result.scores,
                pose_result.class_names,
            )
        )
        for raw_vertices, det_score, class_name in self._pose_nms(detections):
            vertices = np.asarray(raw_vertices, dtype=np.float32).reshape(-1, 2)
            if vertices.shape != (4, 2) or not np.all(np.isfinite(vertices)):
                continue

            # ECPose metadata defines TL, BL, BR, TR, whereas PlateNormalizer
            # consumes TL, TR, BR, BL.
            normalizer_vertices = np.ascontiguousarray(
                vertices[[0, 3, 2, 1]], dtype=np.float32
            )
            normalized = self.plate_normalizer.normalize(image, normalizer_vertices)
            normalized = self._as_bgr(normalized)

            # DBNet otherwise tends to merge with/cut off the plate boundary.
            class_name = str(class_name)
            is_local = class_name.casefold() == "local"
            ocr_image = (
                self._preprocess_local_plate(normalized) if is_local else normalized
            )
            padded, pad = self._pad_for_ocr(ocr_image)
            ocr_result = self.ocr_model(padded)
            # The color mask gives DBNet cleaner text rows on illustrated local
            # plates.  Fixed crops retain the original pixels because masking
            # can remove useful anti-aliasing from small individual glyphs.
            field_candidates = self._recognize_fixed_fields(normalized)

            parsed_result = self.parse_ocr_result(
                ocr_result,
                vertices=vertices,
                detection_score=float(det_score),
                normalized_image=normalized,
                original_image=image,
                class_name=class_name,
                field_candidates=field_candidates,
                padding=pad,
                spatial_fields=(self._LOCAL_SPATIAL_FIELDS if is_local else None),
            )
            # These two fields are mandatory on a Japanese plate.  Requiring
            # both removes background quadrilaterals that pass pose NMS but
            # contain no coherent plate text.
            if parsed_result.class_number and parsed_result.number > 0:
                results.append(parsed_result)

        return results

    def create_result_images(self, results: list[LPDetResult]) -> list[np.ndarray]:
        """Create one display image for each recognized plate.

        Each returned BGR image contains the normalized plate at the top and
        ``"{area} {class_number}\n{kana} {number}"`` on a white area below it.

        Args:
            results: Recognition results returned by :meth:`__call__`.

        Returns:
            Display images in the same order as ``results``.
        """
        return [result.result_image for result in results]

    @classmethod
    def _pose_nms(
        cls,
        detections: list[tuple[list[float], float, str]],
        iou_threshold: float = 0.3,
    ) -> list[tuple[list[float], float, str]]:
        """Remove duplicate pose predictions before the expensive OCR stage."""
        if not isinstance(iou_threshold, (int, float)) or isinstance(
            iou_threshold, bool
        ):
            raise TypeError("iou_threshold must be a number")
        if not np.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        valid_detections = []
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
            valid_detections.append((raw_vertices, score, class_name, polygon))

        kept: list[tuple[list[float], float, str]] = []
        kept_polygons: list[np.ndarray] = []
        for raw_vertices, score, class_name, polygon in sorted(
            valid_detections, key=lambda item: item[1], reverse=True
        ):
            if all(
                cls._polygon_iou(polygon, other) < iou_threshold
                for other in kept_polygons
            ):
                kept.append((raw_vertices, score, class_name))
                kept_polygons.append(polygon)
        return kept

    @staticmethod
    def _polygon_iou(left: np.ndarray, right: np.ndarray) -> float:
        left = cv2.convexHull(np.asarray(left, dtype=np.float32))
        right = cv2.convexHull(np.asarray(right, dtype=np.float32))
        left_area = abs(float(cv2.contourArea(left)))
        right_area = abs(float(cv2.contourArea(right)))
        if left_area <= 0.0 or right_area <= 0.0:
            return 0.0
        intersection_area, _ = cv2.intersectConvexConvex(left, right)
        union_area = left_area + right_area - float(intersection_area)
        return float(intersection_area) / union_area if union_area > 0.0 else 0.0

    def parse_ocr_result(
        self,
        ocr_result: OCRResult,
        *,
        vertices: np.ndarray | None = None,
        detection_score: float = 0.0,
        normalized_image: np.ndarray | None = None,
        original_image: np.ndarray | None = None,
        class_name: str = "",
        field_candidates: dict[str, list[_TextCandidate]] | None = None,
        padding: int = 0,
        spatial_fields: Iterable[str] | None = None,
    ) -> LPDetResult:
        """Convert spatial OCR results into the four plate fields.

        The keyword arguments are optional to keep this method useful for
        inspecting or testing an :class:`OCRResult` independently.
        """
        if not isinstance(padding, int) or isinstance(padding, bool):
            raise TypeError("padding must be an integer")
        if padding < 0:
            raise ValueError("padding must not be negative")
        _validate_bgr_image(ocr_result.image)
        if padding and padding * 2 >= min(ocr_result.image.shape[:2]):
            raise ValueError("padding is too large for the OCR image")

        candidates = {
            "area": [],
            "class_number": [],
            "kana": [],
            "number": [],
        }
        if field_candidates:
            unknown_fields = set(field_candidates).difference(candidates)
            if unknown_fields:
                names = ", ".join(sorted(unknown_fields))
                raise ValueError(f"Unknown candidate fields: {names}")
            for field_name, values in field_candidates.items():
                candidates[field_name].extend(values)

        if spatial_fields is None:
            self._collect_spatial_candidates(ocr_result, candidates, padding=padding)
        else:
            selected_fields = frozenset(spatial_fields)
            unknown_fields = selected_fields.difference(candidates)
            if unknown_fields:
                names = ", ".join(sorted(unknown_fields))
                raise ValueError(f"Unknown spatial fields: {names}")
            spatial_candidates = {field_name: [] for field_name in candidates}
            self._collect_spatial_candidates(
                ocr_result, spatial_candidates, padding=padding
            )
            for field_name in selected_fields:
                candidates[field_name].extend(spatial_candidates[field_name])

        area = self._best_area(candidates["area"])
        class_number = self._best_class_number(candidates["class_number"])
        kana = self._best_kana(candidates["kana"])
        number = self._best_number(candidates["number"])

        plate_image = (
            normalized_image
            if normalized_image is not None
            else self._remove_padding(ocr_result.image, padding)
        )
        source_image = original_image if original_image is not None else plate_image
        result_vertices = (
            np.asarray(vertices, dtype=np.float32).reshape(4, 2)
            if vertices is not None
            else np.zeros((4, 2), dtype=np.float32)
        )
        return LPDetResult(
            vertices=result_vertices,
            score=float(detection_score),
            plate_image=plate_image,
            original_image=source_image,
            class_name=class_name,
            area=area.text,
            area_score=area.score,
            class_number=class_number.text,
            class_number_score=class_number.score,
            kana=kana.text,
            kana_score=kana.score,
            number=int(number.text) if number.text else 0,
            number_score=number.score,
        )

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        _validate_bgr_image(image)

    @staticmethod
    def _read_image(path: str | Path) -> np.ndarray:
        """Read an image without relying on OpenCV's path handling."""
        image_path = Path(path)
        encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size > 0 else None
        if image is None:
            raise ValueError(f"Could not decode image file: {image_path}")
        return image

    @staticmethod
    def _as_bgr(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("normalized image must be a numpy.ndarray")
        if image.ndim == 2:
            if image.size == 0:
                raise ValueError("normalized image must not be empty")
            if image.dtype != np.uint8:
                raise TypeError("normalized image must have dtype uint8")
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            _validate_bgr_image(image)
            return image
        raise ValueError(f"normalized image has an invalid shape: {image.shape}")

    @classmethod
    def _preprocess_local_plate(cls, image: np.ndarray) -> np.ndarray:
        """Keep black-to-dark-green pixels and whiten local-plate artwork."""
        cls._validate_image(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        black = value <= cls._LOCAL_BLACK_VALUE_MAX
        dark_green = (
            (hue >= cls._LOCAL_GREEN_HUE_MIN)
            & (hue <= cls._LOCAL_GREEN_HUE_MAX)
            & (saturation >= cls._LOCAL_GREEN_SATURATION_MIN)
            & (value <= cls._LOCAL_GREEN_VALUE_MAX)
        )
        keep = black | dark_green
        filtered = np.full_like(image, 255)
        filtered[keep] = image[keep]
        return filtered

    @staticmethod
    def _pad_for_ocr(image: np.ndarray) -> tuple[np.ndarray, int]:
        pad = max(8, int(round(image.shape[0] * 0.10)))
        padded = cv2.copyMakeBorder(
            image,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=(127, 127, 127),
        )
        return padded, pad

    @staticmethod
    def _remove_padding(image: np.ndarray, padding: int) -> np.ndarray:
        if padding <= 0:
            return image
        if image.shape[0] <= padding * 2 or image.shape[1] <= padding * 2:
            return image
        return image[padding:-padding, padding:-padding]

    def _recognize_fixed_fields(
        self, image: np.ndarray
    ) -> dict[str, list[_TextCandidate]]:
        """Read layout-defined crops with field-specific CTC masks."""
        candidates: dict[str, list[_TextCandidate]] = {
            field_name: [] for field_name in ("area", "class_number", "kana", "number")
        }
        # Custom/mocked OCR implementations can still use spatial parsing.
        required = ("rec_session", "rec_input_name", "decoder")
        if not all(hasattr(self.ocr_model, name) for name in required):
            return candidates

        height, width = image.shape[:2]
        for field_name, (x1, y1, x2, y2) in self._FIELD_RECTS.items():
            left = max(0, min(width - 1, int(round(x1 * width))))
            top = max(0, min(height - 1, int(round(y1 * height))))
            right = max(left + 1, min(width, int(round(x2 * width))))
            bottom = max(top + 1, min(height, int(round(y2 * height))))
            crop = image[top:bottom, left:right]
            if field_name == "area":
                candidate = self._recognize_area(crop)
            elif field_name == "class_number":
                candidate = self._recognize_class_number(crop)
            else:
                candidate = self._recognize_crop(crop, self._FIELD_FILTERS[field_name])
            if candidate.text:
                candidates[field_name].append(candidate)

        kana_candidate = self._recognize_kana(image)
        if kana_candidate.text:
            candidates["kana"].append(kana_candidate)

        # A joined top-row reading is often stronger than either small crop.
        top_row = image[
            : max(1, int(round(height * 0.41))),
            int(round(width * 0.14)) : int(round(width * 0.85)),
        ]
        joined = self._recognize_crop(top_row, None)
        self._split_top_text(joined, candidates)
        return candidates

    def _recognize_crop(
        self, crop: np.ndarray, filters: list[str] | None
    ) -> _TextCandidate:
        if crop.size == 0:
            return _TextCandidate("", 0.0)
        rec_input = preprocess_rec(crop, target_height=self.ocr_model.rec_target_height)
        rec_outputs = self.ocr_model.rec_session.run(
            None, {self.ocr_model.rec_input_name: rec_input}
        )
        text, score = self.ocr_model.decoder.decode(rec_outputs[0], filters=filters)
        return _TextCandidate(str(text).strip(), float(score))

    def _recognize_area(self, crop: np.ndarray) -> _TextCandidate:
        """Choose the most probable complete issuing-area name with CTC."""
        if crop.size == 0:
            return _TextCandidate("", 0.0)
        rec_input = preprocess_rec(crop, target_height=self.ocr_model.rec_target_height)
        rec_outputs = self.ocr_model.rec_session.run(
            None, {self.ocr_model.rec_input_name: rec_input}
        )[0]
        probabilities = np.asarray(rec_outputs, dtype=np.float64)[0]
        log_probabilities = np.log(np.clip(probabilities, 1e-30, 1.0))
        scores = np.asarray(
            [
                self._ctc_log_probability(log_probabilities, tokens) / len(tokens)
                for _, tokens in self._area_tokens
            ],
            dtype=np.float64,
        )
        best_index = int(np.argmax(scores))
        normalizer = self._logsumexp(scores)
        confidence = float(np.exp(scores[best_index] - normalizer))
        return _TextCandidate(self._area_tokens[best_index][0], confidence)

    def _recognize_class_number(self, crop: np.ndarray) -> _TextCandidate:
        """Recognize a two- or three-character classification number."""
        if crop.size == 0:
            return _TextCandidate("", 0.0)
        rec_input = preprocess_rec(crop, target_height=self.ocr_model.rec_target_height)
        rec_outputs = self.ocr_model.rec_session.run(
            None, {self.ocr_model.rec_input_name: rec_input}
        )[0]
        text, score = self.ocr_model.decoder.decode(
            rec_outputs, filters=self._FIELD_FILTERS["class_number"]
        )
        return _TextCandidate(str(text).strip().upper(), float(score))

    def _single_character_scores(
        self, crop: np.ndarray, allowed_characters: Iterable[str]
    ) -> dict[str, float]:
        if crop.size == 0:
            return {}
        rec_input = preprocess_rec(crop, target_height=self.ocr_model.rec_target_height)
        rec_outputs = self.ocr_model.rec_session.run(
            None, {self.ocr_model.rec_input_name: rec_input}
        )[0]
        probabilities = np.asarray(rec_outputs)[0]
        character_indices = {
            character: self.ocr_model.decoder.character.index(character)
            for character in allowed_characters
            if character in self.ocr_model.decoder.character
        }
        if not character_indices:
            return {}

        return {
            character: max(0.0, float(np.max(probabilities[:, index])))
            for character, index in character_indices.items()
        }

    def _recognize_kana(self, image: np.ndarray) -> _TextCandidate:
        height, width = image.shape[:2]
        combined = {character: 0.0 for character in self.kana_characters}
        for rect, weight in self._KANA_RECTS:
            x1, y1, x2, y2 = rect
            left = max(0, min(width - 1, int(round(x1 * width))))
            top = max(0, min(height - 1, int(round(y1 * height))))
            right = max(left + 1, min(width, int(round(x2 * width))))
            bottom = max(top + 1, min(height, int(round(y2 * height))))
            scores = self._single_character_scores(
                image[top:bottom, left:right], self.kana_characters
            )
            score_sum = sum(scores.values())
            if score_sum <= 0.0:
                continue
            for character, score in scores.items():
                combined[character] += weight * score / score_sum

        if not combined:
            return _TextCandidate("", 0.0)
        character = max(combined, key=combined.get)
        score_sum = sum(combined.values())
        confidence = combined[character] / score_sum if score_sum > 0.0 else 0.0
        return _TextCandidate(character, float(confidence))

    def _collect_spatial_candidates(
        self,
        ocr_result: OCRResult,
        candidates: dict[str, list[_TextCandidate]],
        *,
        padding: int,
    ) -> None:
        image_height, image_width = ocr_result.image.shape[:2]
        plate_height = max(1, image_height - padding * 2)
        plate_width = max(1, image_width - padding * 2)

        for box, text, score in zip(
            ocr_result.boxes, ocr_result.texts, ocr_result.scores
        ):
            text = str(text).strip()
            if not text:
                continue
            try:
                points = np.asarray(box, dtype=np.float32).reshape(4, 2)
            except (TypeError, ValueError):
                continue
            if not np.all(np.isfinite(points)):
                continue
            center_x = (float(np.mean(points[:, 0])) - padding) / plate_width
            center_y = (float(np.mean(points[:, 1])) - padding) / plate_height
            candidate = _TextCandidate(text, float(score))

            if center_y < 0.48:
                # Use position for separately detected fields, then also try a
                # lexical split because DBNet commonly merges the top row.
                if center_x < 0.52:
                    candidates["area"].append(candidate)
                else:
                    candidates["class_number"].append(candidate)
                self._split_top_text(candidate, candidates)
            else:
                self._split_bottom_text(candidate, candidates)

    def _split_top_text(
        self,
        candidate: _TextCandidate,
        candidates: dict[str, list[_TextCandidate]],
    ) -> None:
        compact = re.sub(r"\s+", "", candidate.text).upper()
        match = re.search(r"([0-9A-Z]{2,3})$", compact)
        if match:
            class_text = match.group(1)
            area_text = compact[: match.start()]
            if area_text:
                candidates["area"].append(_TextCandidate(area_text, candidate.score))
            candidates["class_number"].append(
                _TextCandidate(class_text, candidate.score)
            )
            return

        # A field-specific area decoder removes digits.  Keeping the joined
        # reading here still lets exact area substrings beat a weak crop.
        for area in self.areas:
            if compact.startswith(area):
                candidates["area"].append(_TextCandidate(area, candidate.score))
                suffix = compact[len(area) :]
                if suffix:
                    candidates["class_number"].append(
                        _TextCandidate(suffix, candidate.score)
                    )
                break

    def _split_bottom_text(
        self,
        candidate: _TextCandidate,
        candidates: dict[str, list[_TextCandidate]],
    ) -> None:
        kana_text = "".join(
            character for character in candidate.text if character in self.hiragana
        )
        if kana_text:
            candidates["kana"].append(_TextCandidate(kana_text[0], candidate.score))
        if any(character.isdigit() for character in candidate.text):
            candidates["number"].append(candidate)

    def _best_area(self, candidates: Iterable[_TextCandidate]) -> _TextCandidate:
        best = _TextCandidate("", 0.0)
        for candidate in candidates:
            text = re.sub(r"[\s・·\-0-9A-Z]", "", candidate.text.upper())
            if not text:
                continue
            if text in self.areas:
                mapped = text
                similarity = 1.0
            else:
                mapped, distance = min(
                    ((area, self._edit_distance(text, area)) for area in self.areas),
                    key=lambda item: (item[1], abs(len(item[0]) - len(text))),
                )
                similarity = max(0.0, 1.0 - distance / max(len(text), len(mapped)))
                # Do not turn unrelated noise into a plausible area.
                if similarity < 0.45:
                    continue
            score = float(candidate.score) * similarity
            if score > best.score:
                best = _TextCandidate(mapped, score)
        return best

    def _best_class_number(
        self, candidates: Iterable[_TextCandidate]
    ) -> _TextCandidate:
        valid_candidates: list[_TextCandidate] = []
        for candidate in candidates:
            text = self._ALNUM_RE.sub("", candidate.text.upper())
            if len(text) not in (2, 3):
                continue
            if not text[0].isdigit():
                continue
            if len(text) == 2 and not text[1].isdigit():
                continue
            if len(text) == 3 and not text[1].isdigit():
                continue
            if len(text) == 3 and text[2] not in self._CLASS_SUFFIX_CHARACTERS:
                continue
            valid_candidates.append(_TextCandidate(text, float(candidate.score)))
        if not valid_candidates:
            return _TextCandidate("", 0.0)

        return max(valid_candidates, key=lambda value: value.score)

    def _best_kana(self, candidates: Iterable[_TextCandidate]) -> _TextCandidate:
        best = _TextCandidate("", 0.0)
        for candidate in candidates:
            text = next(
                (
                    character
                    for character in candidate.text
                    if character in self.hiragana
                ),
                "",
            )
            if text and candidate.score > best.score:
                best = _TextCandidate(text, float(candidate.score))
        return best

    def _best_number(self, candidates: Iterable[_TextCandidate]) -> _TextCandidate:
        best = _TextCandidate("", 0.0)
        for candidate in candidates:
            digits = "".join(self._DIGIT_RE.findall(candidate.text))
            if not 1 <= len(digits) <= 4:
                continue
            value = int(digits)
            if not 1 <= value <= 9999:
                continue
            if candidate.score > best.score:
                best = _TextCandidate(str(value), float(candidate.score))
        return best

    @staticmethod
    def _edit_distance(left: str, right: str) -> int:
        """Small dependency-free Levenshtein distance for area correction."""
        if len(left) < len(right):
            left, right = right, left
        previous = list(range(len(right) + 1))
        for row, left_character in enumerate(left, start=1):
            current = [row]
            for column, right_character in enumerate(right, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[column] + 1,
                        previous[column - 1] + (left_character != right_character),
                    )
                )
            previous = current
        return previous[-1]

    @staticmethod
    def _logsumexp(values: np.ndarray) -> float:
        maximum = float(np.max(values))
        if not np.isfinite(maximum):
            return -np.inf
        return maximum + float(np.log(np.exp(values - maximum).sum()))

    @classmethod
    def _ctc_log_probability(
        cls, log_probabilities: np.ndarray, tokens: tuple[int, ...]
    ) -> float:
        """CTC forward score for one vocabulary entry (blank index is zero)."""
        extended = [0]
        for token in tokens:
            extended.extend((token, 0))

        previous = np.full(len(extended), -np.inf, dtype=np.float64)
        previous[0] = log_probabilities[0, 0]
        previous[1] = log_probabilities[0, extended[1]]
        for timestep in range(1, len(log_probabilities)):
            current = np.full_like(previous, -np.inf)
            for state, token in enumerate(extended):
                incoming = [previous[state]]
                if state > 0:
                    incoming.append(previous[state - 1])
                if state > 1 and token != 0 and token != extended[state - 2]:
                    incoming.append(previous[state - 2])
                current[state] = cls._logsumexp(np.asarray(incoming))
                current[state] += log_probabilities[timestep, token]
            previous = current
        return cls._logsumexp(previous[-2:])
