"""Runtime-only implementation of the frozen Traditional Quality Ranker V4."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage


WEIGHTS = {
    "segmentation": 0.40,
    "individual_clarity": 0.22,
    "morphology_distribution": 0.14,
    "gray_visibility": 0.10,
    "focus": 0.06,
    "background_illumination": 0.06,
    "acquisition_spec": 0.02,
}
TARGET_COUNT = 46
IMAGE_SIZE = 384
VERSION = "traditional_quality_ranker_v4_hierarchical_46"
IDENTITY = [
    "image_id", "relative_path", "path", "source", "case_id", "sha256",
    "near_duplicate_group",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_gray(path: Path) -> tuple[np.ndarray, int, int]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        width, height = image.size
        image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        gray = np.asarray(image, dtype=np.float32) / 255.0
    return gray, width, height


def object_features(path: Path) -> dict[str, float]:
    """Compute the V4 connected-component measurements for one image."""
    gray, _, _ = _load_gray(path)
    pixel_count = gray.size
    background = ndimage.uniform_filter(
        ndimage.maximum_filter(gray, size=41, mode="nearest"), size=21, mode="nearest"
    )
    residual = np.maximum(background - gray, 0.0)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    threshold = max(
        median + 3.0 * mad,
        float(np.quantile(residual, 0.75)) + 0.008,
        0.015,
    )
    mask = residual > threshold
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2), dtype=bool))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    areas = np.bincount(labels.ravel(), minlength=count + 1).astype(np.float64)
    min_area = max(18.0, pixel_count * 0.00035)
    max_area = pixel_count * 0.035

    eligible_labels: list[int] = []
    ranges: list[float] = []
    stds: list[float] = []
    gradients: list[float] = []
    centroids: list[tuple[float, float]] = []
    eligible_areas: list[float] = []
    gy, gx = np.gradient(gray)
    gradient = np.hypot(gx, gy)
    objects = ndimage.find_objects(labels)
    for label_id, slices in enumerate(objects, start=1):
        if slices is None or not (min_area <= areas[label_id] <= max_area):
            continue
        height = slices[0].stop - slices[0].start
        width = slices[1].stop - slices[1].start
        aspect = max(height, width) / max(min(height, width), 1)
        fill = areas[label_id] / max(height * width, 1)
        if aspect > 9.0 or fill < 0.08:
            continue
        component = labels[slices] == label_id
        values = gray[slices][component]
        grad_values = gradient[slices][component]
        if values.size == 0:
            continue
        eligible_labels.append(label_id)
        eligible_areas.append(float(areas[label_id]))
        ranges.append(float(np.quantile(values, 0.90) - np.quantile(values, 0.10)))
        stds.append(float(np.std(values)))
        gradients.append(float(np.median(grad_values)))
        coords = np.argwhere(labels == label_id)
        centroids.append((float(coords[:, 0].mean()), float(coords[:, 1].mean())))

    foreground_pixels = max(float(mask.sum()), 1.0)
    eligible_pixels = float(sum(eligible_areas))
    artifact_fraction = float(
        areas[1:][areas[1:] > max_area].sum() / foreground_pixels
    ) if count else 0.0
    usable_fraction = min(eligible_pixels / foreground_pixels, 1.0)
    estimated_count = len(eligible_labels)
    count_closeness = math.exp(-abs(estimated_count - TARGET_COUNT) / 9.0)
    segmentation_raw = (
        0.75 * count_closeness
        + 0.15 * usable_fraction
        + 0.10 * (1.0 - min(artifact_fraction, 1.0))
    )
    if eligible_areas:
        area_array = np.asarray(eligible_areas)
        area_cv = float(area_array.std() / max(area_array.mean(), 1e-8))
        individual_clarity_raw = (
            0.60 * float(np.median(ranges))
            + 0.30 * float(np.median(stds))
            + 0.10 * float(np.median(gradients))
        )
    else:
        area_cv = 9.0
        individual_clarity_raw = 0.0
    if len(centroids) >= 2:
        points = np.asarray(centroids)
        spatial_dispersion = float(
            np.sqrt(np.var(points[:, 0]) + np.var(points[:, 1]))
            / max(math.sqrt(gray.shape[0] ** 2 + gray.shape[1] ** 2), 1.0)
        )
    else:
        spatial_dispersion = 0.0

    return {
        "estimated_chromosome_count": float(estimated_count),
        "count_absolute_error": float(abs(estimated_count - TARGET_COUNT)),
        "count_closeness": count_closeness,
        "usable_component_fraction": usable_fraction,
        "large_artifact_fraction": artifact_fraction,
        "eligible_component_area_cv": area_cv,
        "object_spatial_dispersion": spatial_dispersion,
        "single_object_gray_range_median": float(np.median(ranges)) if ranges else 0.0,
        "single_object_gray_std_median": float(np.median(stds)) if stds else 0.0,
        "single_object_gradient_median": float(np.median(gradients)) if gradients else 0.0,
        "segmentation_raw": segmentation_raw,
        "individual_clarity_raw": individual_clarity_raw,
    }


def object_feature_worker(item: tuple[str, str]) -> dict[str, float | str]:
    sha256, path = item
    result: dict[str, float | str] = object_features(Path(path))
    result["sha256"] = sha256
    return result


def _add_category_raw(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    eps = 1e-8
    out["morphology_distribution_raw"] = (
        1.8 * out["usable_component_fraction"]
        + 0.8 * out["object_spatial_dispersion"]
        + 0.18 * out["grid_entropy"]
        - 0.30 * np.log1p(out["eligible_component_area_cv"].clip(0, 9))
        - 0.70 * out["large_artifact_fraction"].clip(0, 1)
        - 0.25 * out["border_foreground_fraction"].clip(0, 1)
        - 0.15 * out["hole_fraction"].clip(0, 1)
    )
    out["gray_visibility_raw"] = (
        0.55 * np.log1p(out["cnr"].clip(lower=0))
        + 1.8 * out["foreground_background_gap"].clip(lower=0)
        + 0.7 * out["intensity_dynamic_p95_p05"].clip(lower=0)
        - 1.2 * out["dark_saturation_fraction"].clip(0, 1)
        - 1.2 * out["bright_saturation_fraction"].clip(0, 1)
    )
    out["focus_raw"] = (
        0.45 * np.log1p(5000 * out["laplacian_variance_foreground"].clip(lower=0))
        + 0.35 * np.log1p(2000 * out["tenengrad_foreground"].clip(lower=0))
        + 0.20 * np.log1p(100 * out["high_frequency_foreground"].clip(lower=0))
    )
    out["background_illumination_raw"] = -(
        2.5 * out["background_std"].clip(lower=0)
        + 4.0 * out["background_field_std"].clip(lower=0)
        + 8.0 * out["background_high_frequency"].clip(lower=0)
    )
    out["acquisition_spec_raw"] = (
        np.log1p(out["original_megapixels"].clip(lower=0))
        - 0.08 * np.abs(np.log(out["original_aspect_ratio"].clip(lower=eps)))
    )
    return out


def _fixed_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    ranks = np.searchsorted(reference, np.asarray(values, dtype=np.float64), side="right")
    return 100.0 * ranks / len(reference)


def score_frame(frame: pd.DataFrame, references: dict[str, np.ndarray]) -> pd.DataFrame:
    out = _add_category_raw(frame)
    for category, weight in WEIGHTS.items():
        out[f"{category}_score"] = _fixed_percentile(
            out[f"{category}_raw"].to_numpy(), references[category]
        )
        out[f"{category}_contribution"] = weight * out[f"{category}_score"]
    out["score"] = sum(out[f"{category}_contribution"] for category in WEIGHTS)
    return out


def load_ranker(model_path: Path) -> dict:
    bundle = joblib.load(model_path)
    if bundle.get("version") != VERSION:
        raise ValueError(f"Expected {VERSION}, found {bundle.get('version')!r}")
    if bundle.get("weights") != WEIGHTS:
        raise ValueError("Model weights differ from this runtime")
    if bundle.get("target_chromosome_count") != TARGET_COUNT:
        raise ValueError("Model target count differs from this runtime")
    return bundle
