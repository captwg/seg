#!/usr/bin/env python3
"""Build and run the fixed-reference hierarchical chromosome quality ranker.

This intentionally remains a no-GT, traditional-image-analysis ranker.  Its
highest-weight proxy is how close the connected-component estimate is to 46.
"""
from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
V3 = WORKSPACE / "traditional_quality_ranker_v3_fixed_calibration"
SPLIT_PATH = V3 / "lineage" / "mixed_train_test_split.csv"
V3_FEATURE_DIR = V3 / "legacy" / "v1" / "features"
BATCH_FEATURE_DIR = V3 / "example_20260730_scoring" / "features"

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


def load_gray(path: Path) -> tuple[np.ndarray, int, int]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        width, height = image.size
        image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        gray = np.asarray(image, dtype=np.float32) / 255.0
    return gray, width, height


def object_features(path: Path) -> dict[str, float]:
    gray, _, _ = load_gray(path)
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


def add_category_raw(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    eps = 1e-8
    # All category raw values follow the same convention: larger is better.
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


def fixed_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    ranks = np.searchsorted(reference, np.asarray(values, dtype=np.float64), side="right")
    return 100.0 * ranks / len(reference)


def score_frame(frame: pd.DataFrame, references: dict[str, np.ndarray]) -> pd.DataFrame:
    out = add_category_raw(frame)
    for category in WEIGHTS:
        raw_name = f"{category}_raw"
        out[f"{category}_score"] = fixed_percentile(out[raw_name].to_numpy(), references[category])
        out[f"{category}_contribution"] = WEIGHTS[category] * out[f"{category}_score"]
    out["score"] = sum(out[f"{category}_contribution"] for category in WEIGHTS)
    return out


def extract_objects(frame: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    if cache_path.is_file():
        cached = pd.read_parquet(cache_path)
        if len(cached) == len(frame) and set(cached.sha256) == set(frame.sha256):
            return frame.merge(cached.drop(columns=[c for c in IDENTITY if c != "sha256" and c in cached]), on="sha256", validate="one_to_one")
    rows: list[dict] = []
    total = len(frame)
    inputs = [(str(row.sha256), str(row.path)) for row in frame.itertuples(index=False)]
    with ProcessPoolExecutor(max_workers=8) as executor:
        for index, metrics in enumerate(executor.map(object_feature_worker, inputs, chunksize=8), start=1):
            rows.append(metrics)
            if index % 250 == 0 or index == total:
                print(f"object_features {index}/{total}", flush=True)
    cached = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(cache_path, index=False)
    return frame.merge(cached, on="sha256", validate="one_to_one")


def load_original_features(paths: list[Path]) -> pd.DataFrame:
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if "family" in frame:
        frame = frame[frame["family"] == "original"].copy()
    if frame.sha256.duplicated().any():
        raise RuntimeError("Duplicate SHA256 values in original feature input")
    return frame.reset_index(drop=True)


def output_columns(frame: pd.DataFrame, rank_name: str) -> pd.DataFrame:
    detail = [
        "estimated_chromosome_count", "count_absolute_error",
        "single_object_gray_range_median", "single_object_gray_std_median",
    ]
    category = []
    for name in WEIGHTS:
        category.extend([f"{name}_score", f"{name}_contribution"])
    columns = [rank_name, "score"] + detail + category + IDENTITY
    return frame[columns].copy()


def main() -> int:
    old_features = load_original_features(sorted(V3_FEATURE_DIR.glob("quality_features_shard*.parquet")))
    split = pd.read_csv(SPLIT_PATH)[["image_id", "split"]]
    all_frame = old_features.merge(split, on="image_id", validate="one_to_one")
    all_frame = extract_objects(all_frame, ROOT / "calibration" / "all_5726_object_features.parquet")
    all_frame = add_category_raw(all_frame)
    training = all_frame[all_frame.split == "train"].copy()
    heldout = all_frame[all_frame.split == "test"].copy()
    references = {
        name: np.sort(training[f"{name}_raw"].to_numpy(dtype=np.float64))
        for name in WEIGHTS
    }
    bundle = {
        "version": "traditional_quality_ranker_v4_hierarchical_46",
        "target_chromosome_count": TARGET_COUNT,
        "image_size": IMAGE_SIZE,
        "weights": WEIGHTS,
        "training_reference_count": len(training),
        "references": references,
        "score_definition": "weighted sum of seven training-reference fixed ECDF component scores",
        "warning": "No-GT traditional proxy score; not Dice, AP, or segmentation success probability.",
    }
    model_path = ROOT / "models" / "hierarchical_46_fixed_ranker.joblib"
    joblib.dump(bundle, model_path, compress=3)

    heldout = score_frame(heldout, references)
    heldout = heldout.sort_values(["score", "sha256"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    heldout.insert(0, "rank_in_test", np.arange(1, len(heldout) + 1))
    heldout_output = output_columns(heldout, "rank_in_test")
    heldout_path = ROOT / "scores" / "heldout_test_scores.csv"
    heldout_output.to_csv(heldout_path, index=False, encoding="utf-8")

    batch = load_original_features(sorted(BATCH_FEATURE_DIR.glob("original_features_shard*.parquet")))
    batch = extract_objects(batch, ROOT / "calibration" / "dataset_20260730_object_features.parquet")
    batch = score_frame(batch, references)
    batch = batch.sort_values(["score", "sha256"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    batch.insert(0, "rank_in_batch", np.arange(1, len(batch) + 1))
    split_sha = pd.read_csv(SPLIT_PATH)[["sha256", "split"]]
    batch = batch.merge(split_sha, on="sha256", how="left", validate="one_to_one")
    batch_output = output_columns(batch, "rank_in_batch")
    batch_output["model_build_split"] = batch["split"].fillna("external")
    batch_path = ROOT / "scores" / "dataset_20260730_scores.csv"
    batch_output.to_csv(batch_path, index=False, encoding="utf-8")

    spec = {
        "status": "PASS",
        "target_count": TARGET_COUNT,
        "weights": WEIGHTS,
        "training_reference_images": len(training),
        "heldout_images": len(heldout),
        "dataset_20260730_images": len(batch),
        "model_sha256": sha256_file(model_path),
        "heldout_scores_sha256": sha256_file(heldout_path),
        "dataset_20260730_scores_sha256": sha256_file(batch_path),
        "gates": {
            "weights_sum_to_one": abs(sum(WEIGHTS.values()) - 1.0) < 1e-12,
            "priority_is_strictly_ordered_except_requested_tie": list(WEIGHTS.values()) == [0.40, 0.22, 0.14, 0.10, 0.06, 0.06, 0.02],
            "training_reference_count_4455": len(training) == 4455,
            "heldout_count_1271": len(heldout) == 1271,
            "dataset_20260730_count_60": len(batch) == 60,
            "dataset_20260730_split_is_48_train_12_test": batch["split"].value_counts().to_dict() == {"train": 48, "test": 12},
            "score_equals_sum_of_weighted_components": bool(
                np.allclose(
                    heldout["score"],
                    sum(heldout[f"{name}_contribution"] for name in WEIGHTS),
                    atol=1e-12,
                    rtol=0.0,
                )
            ),
            "all_scores_finite_0_100": bool(
                np.isfinite(heldout.score).all() and heldout.score.between(0, 100).all()
                and np.isfinite(batch.score).all() and batch.score.between(0, 100).all()
            ),
        },
    }
    spec["status"] = "PASS" if all(spec["gates"].values()) else "FAIL"
    (ROOT / "reports" / "build_validation.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(spec, ensure_ascii=False, indent=2))
    return 0 if spec["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
