from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from scipy import ndimage


DEGRADATION_FAMILIES = (
    "contrast",
    "illumination",
    "exposure",
    "blur",
    "noise",
    "contamination",
    "crowding",
)

BENIGN_VARIANTS = ("hflip", "rot180")

SUBSCORE_FAMILIES = {
    "visibility": ("contrast", "illumination", "exposure"),
    "focus": ("blur",),
    "clean": ("noise", "contamination"),
    "metaphase": ("crowding",),
    "overall": DEGRADATION_FAMILIES,
}

IDENTITY_COLUMNS = (
    "image_id",
    "path",
    "relative_path",
    "source",
    "case_id",
    "sha256",
    "near_duplicate_group",
    "variant",
    "family",
    "level",
    "severity",
    "shard_index",
)


@dataclass
class LoadedImage:
    gray: torch.Tensor
    valid: torch.Tensor
    width: int
    height: int


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def stable_seed(*parts: str | int) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFFFFFF


def load_image(path: Path, image_size: int) -> LoadedImage:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        width, height = image.size
        image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        resized = np.asarray(image, dtype=np.float32) / 255.0
    border = np.concatenate((resized[0], resized[-1], resized[:, 0], resized[:, -1]))
    pad_value = float(np.quantile(border, 0.75)) if border.size else 1.0
    canvas = np.full((image_size, image_size), pad_value, dtype=np.float32)
    valid = np.zeros((image_size, image_size), dtype=np.float32)
    top = (image_size - resized.shape[0]) // 2
    left = (image_size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    valid[top : top + resized.shape[0], left : left + resized.shape[1]] = 1.0
    return LoadedImage(
        gray=torch.from_numpy(canvas).unsqueeze(0),
        valid=torch.from_numpy(valid).unsqueeze(0),
        width=width,
        height=height,
    )


def _nanquantiles(value: torch.Tensor, valid: torch.Tensor, quantiles: Iterable[float]) -> torch.Tensor:
    flattened = value.flatten(1)
    mask = valid.flatten(1).bool()
    masked = torch.where(mask, flattened, torch.full_like(flattened, torch.nan))
    q = torch.tensor(tuple(quantiles), device=value.device, dtype=value.dtype)
    return torch.nanquantile(masked, q, dim=1).transpose(0, 1)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numerator = (value * mask).flatten(1).sum(1)
    denominator = mask.flatten(1).sum(1).clamp_min(1.0)
    return numerator / denominator


def _masked_var(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = _masked_mean(value, mask)
    centered = (value - mean[:, None, None, None]).square()
    return _masked_mean(centered, mask)


def _gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d[None, None]


def gaussian_blur(value: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = _gaussian_kernel(sigma, value.device, value.dtype)
    radius = kernel.shape[-1] // 2
    return F.conv2d(value, kernel, padding=radius)


def apply_degradation(
    gray: torch.Tensor,
    valid: torch.Tensor,
    family: str,
    level: int,
    seed: int,
) -> torch.Tensor:
    if level <= 0:
        return gray
    strength = float(level) / 3.0
    border_mean = _masked_mean(gray, valid)[:, None, None, None]
    if family == "contrast":
        factor = {1: 0.78, 2: 0.56, 3: 0.36}[level]
        output = border_mean + factor * (gray - border_mean)
    elif family == "illumination":
        size = gray.shape[-1]
        axis = torch.linspace(-1.0, 1.0, size, device=gray.device, dtype=gray.dtype)
        field = (axis[None, None, None, :] + 0.65 * axis[None, None, :, None]) / 1.65
        amplitude = {1: 0.10, 2: 0.22, 3: 0.36}[level]
        output = gray + amplitude * field
    elif family == "exposure":
        amount = {1: 0.10, 2: 0.22, 3: 0.36}[level]
        output = gray * (1.0 - amount) + amount
    elif family == "blur":
        sigma = {1: 1.1, 2: 2.2, 3: 3.8}[level]
        output = gaussian_blur(gray, sigma)
    elif family == "noise":
        generator = torch.Generator(device=gray.device)
        generator.manual_seed(seed)
        sigma = {1: 0.012, 2: 0.032, 3: 0.065}[level]
        noise = torch.randn(gray.shape, generator=generator, device=gray.device, dtype=gray.dtype)
        output = gray + sigma * noise
    elif family == "contamination":
        generator = torch.Generator(device=gray.device)
        generator.manual_seed(seed)
        batch, _, height, width = gray.shape
        yy = torch.arange(height, device=gray.device, dtype=gray.dtype)[None, :, None]
        xx = torch.arange(width, device=gray.device, dtype=gray.dtype)[None, None, :]
        contamination = torch.zeros_like(gray)
        count = level + 1
        for _ in range(count):
            center_y = torch.randint(int(0.08 * height), int(0.92 * height), (batch,), generator=generator, device=gray.device)
            center_x = torch.randint(int(0.08 * width), int(0.92 * width), (batch,), generator=generator, device=gray.device)
            radius = torch.randint(max(3, width // 45), max(4, width // 15), (batch,), generator=generator, device=gray.device)
            distance = (yy - center_y[:, None, None]).square() + (xx - center_x[:, None, None]).square()
            blob = torch.exp(-distance / (2.0 * radius[:, None, None].float().square()))[:, None]
            contamination = torch.maximum(contamination, blob)
        amplitude = {1: 0.18, 2: 0.32, 3: 0.48}[level]
        output = gray - amplitude * contamination
    elif family == "crowding":
        background = F.avg_pool2d(F.max_pool2d(gray, 31, stride=1, padding=15), 15, stride=1, padding=7)
        darkness = (background - gray).clamp_min(0.0) * valid
        shifts = ((19, -23), (-27, -14), (11, 31))
        output = gray.clone()
        for shift_y, shift_x in shifts[:level]:
            shifted = torch.roll(darkness, shifts=(shift_y, shift_x), dims=(-2, -1))
            output = output - 0.62 * shifted
    else:
        raise ValueError(f"Unknown degradation family: {family}")
    return output.clamp(0.0, 1.0) * valid + gray * (1.0 - valid)


def _morphological_skeleton(mask: np.ndarray, max_iterations: int = 28) -> np.ndarray:
    structure = ndimage.generate_binary_structure(2, 1)
    current = mask.astype(bool, copy=True)
    skeleton = np.zeros_like(current)
    for _ in range(max_iterations):
        if not current.any():
            break
        eroded = ndimage.binary_erosion(current, structure=structure)
        opened = ndimage.binary_dilation(eroded, structure=structure)
        skeleton |= current & ~opened
        current = eroded
    return skeleton


def morphology_features(mask: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    mask = mask.astype(bool) & valid.astype(bool)
    valid_pixels = int(valid.sum())
    foreground_pixels = int(mask.sum())
    if valid_pixels <= 0:
        return {name: 0.0 for name in MORPH_FEATURE_NAMES}

    structure = ndimage.generate_binary_structure(2, 2)
    cleaned = ndimage.binary_opening(mask, structure=structure, iterations=1)
    labels, component_count = ndimage.label(cleaned, structure=structure)
    areas = np.bincount(labels.ravel())[1:].astype(np.float64)
    foreground_for_components = max(float(areas.sum()), 1.0)
    small_threshold = max(3.0, valid_pixels * 0.00045)
    small = areas < small_threshold
    largest = float(areas.max()) if areas.size else 0.0

    border = np.zeros_like(mask)
    border[:2] = True
    border[-2:] = True
    border[:, :2] = True
    border[:, -2:] = True
    border &= valid.astype(bool)
    border_foreground = float((mask & border).sum())

    large_round_count = 0
    elongated_count = 0
    objects = ndimage.find_objects(labels)
    for label_index, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        height = slices[0].stop - slices[0].start
        width = slices[1].stop - slices[1].start
        area = float(areas[label_index - 1])
        aspect = max(height, width) / max(min(height, width), 1)
        fill = area / max(float(height * width), 1.0)
        if area >= valid_pixels * 0.008 and aspect <= 1.65 and fill >= 0.42:
            large_round_count += 1
        if area >= valid_pixels * 0.0008 and aspect >= 5.5:
            elongated_count += 1

    coordinates = np.argwhere(mask)
    if coordinates.size:
        y_dispersion = float(np.std(coordinates[:, 0]) / max(mask.shape[0], 1))
        x_dispersion = float(np.std(coordinates[:, 1]) / max(mask.shape[1], 1))
        dispersion = math.sqrt(x_dispersion * x_dispersion + y_dispersion * y_dispersion)
    else:
        dispersion = 0.0

    grid_mass = []
    for y_index in range(4):
        for x_index in range(4):
            y0 = y_index * mask.shape[0] // 4
            y1 = (y_index + 1) * mask.shape[0] // 4
            x0 = x_index * mask.shape[1] // 4
            x1 = (x_index + 1) * mask.shape[1] // 4
            grid_mass.append(float(mask[y0:y1, x0:x1].sum()))
    grid_mass_array = np.asarray(grid_mass, dtype=np.float64)
    if grid_mass_array.sum() > 0:
        probabilities = grid_mass_array / grid_mass_array.sum()
        nonzero = probabilities > 0
        grid_entropy = float(-(probabilities[nonzero] * np.log(probabilities[nonzero])).sum() / np.log(16.0))
    else:
        grid_entropy = 0.0

    small_mask = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((72, 72), Image.Resampling.NEAREST)
    ) > 0
    skeleton = _morphological_skeleton(small_mask)
    neighbor_kernel = np.ones((3, 3), dtype=np.int16)
    neighbors = ndimage.convolve(skeleton.astype(np.int16), neighbor_kernel, mode="constant") - skeleton
    skeleton_pixels = max(int(skeleton.sum()), 1)
    branch_density = float(((skeleton) & (neighbors >= 3)).sum() / skeleton_pixels)
    endpoint_density = float(((skeleton) & (neighbors == 1)).sum() / skeleton_pixels)

    filled = ndimage.binary_fill_holes(cleaned)
    hole_fraction = float((filled.sum() - cleaned.sum()) / foreground_for_components)
    return {
        "component_count": float(component_count),
        "component_count_per_mpix": float(component_count * 1_000_000.0 / valid_pixels),
        "small_component_count": float(small.sum()),
        "small_component_area_fraction": float(areas[small].sum() / foreground_for_components) if areas.size else 0.0,
        "largest_component_fraction": largest / foreground_for_components,
        "border_foreground_fraction": border_foreground / max(float(foreground_pixels), 1.0),
        "large_round_count": float(large_round_count),
        "elongated_component_count": float(elongated_count),
        "foreground_dispersion": dispersion,
        "grid_entropy": grid_entropy,
        "skeleton_branch_density": branch_density,
        "skeleton_endpoint_density": endpoint_density,
        "hole_fraction": hole_fraction,
    }


MORPH_FEATURE_NAMES = (
    "component_count",
    "component_count_per_mpix",
    "small_component_count",
    "small_component_area_fraction",
    "largest_component_fraction",
    "border_foreground_fraction",
    "large_round_count",
    "elongated_component_count",
    "foreground_dispersion",
    "grid_entropy",
    "skeleton_branch_density",
    "skeleton_endpoint_density",
    "hole_fraction",
)


def extract_batch_features(
    gray: torch.Tensor,
    valid: torch.Tensor,
    original_sizes: list[tuple[int, int]],
    morphology_size: int = 128,
) -> list[dict[str, float]]:
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=gray.device,
        dtype=gray.dtype,
    )[None, None] / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    laplace_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=gray.device,
        dtype=gray.dtype,
    )[None, None]

    background = F.avg_pool2d(F.max_pool2d(gray, 41, stride=1, padding=20), 21, stride=1, padding=10)
    residual = (background - gray).clamp_min(0.0)
    intensity_q = _nanquantiles(gray, valid, (0.01, 0.05, 0.50, 0.95, 0.99))
    residual_q = _nanquantiles(residual, valid, (0.50, 0.75, 0.90, 0.95, 0.99))
    residual_median = residual_q[:, 0]
    residual_abs_dev = (residual - residual_median[:, None, None, None]).abs()
    residual_mad = _nanquantiles(residual_abs_dev, valid, (0.50,))[:, 0]
    threshold = torch.maximum(
        residual_median + 3.0 * residual_mad,
        torch.maximum(residual_q[:, 1] + 0.008, torch.full_like(residual_median, 0.015)),
    )
    foreground = (residual > threshold[:, None, None, None]).float() * valid
    foreground_dilated = (F.max_pool2d(foreground, 7, stride=1, padding=3) > 0).float() * valid
    background_mask = (1.0 - foreground_dilated) * valid

    gradient_x = F.conv2d(gray, sobel_x, padding=1)
    gradient_y = F.conv2d(gray, sobel_y, padding=1)
    gradient = torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-12)
    laplace = F.conv2d(gray, laplace_kernel, padding=1)
    high_frequency = (gray - gaussian_blur(gray, 1.3)).abs()
    brenner_x = (gray[:, :, :, 2:] - gray[:, :, :, :-2]).square()
    brenner_y = (gray[:, :, 2:, :] - gray[:, :, :-2, :]).square()
    brenner_mask_x = valid[:, :, :, 2:] * valid[:, :, :, :-2]
    brenner_mask_y = valid[:, :, 2:, :] * valid[:, :, :-2, :]
    gradient_q = _nanquantiles(gradient, valid, (0.90, 0.95, 0.99))

    fg_mean = _masked_mean(gray, foreground)
    bg_mean = _masked_mean(gray, background_mask)
    fg_var = _masked_var(gray, foreground)
    bg_var = _masked_var(gray, background_mask)
    gap = (bg_mean - fg_mean).clamp_min(0.0)
    cnr = gap / torch.sqrt(0.5 * (fg_var + bg_var) + 1e-8)

    scalar = {
        "intensity_p01": intensity_q[:, 0],
        "intensity_p05": intensity_q[:, 1],
        "intensity_p50": intensity_q[:, 2],
        "intensity_p95": intensity_q[:, 3],
        "intensity_p99": intensity_q[:, 4],
        "intensity_dynamic_p95_p05": intensity_q[:, 3] - intensity_q[:, 1],
        "dark_saturation_fraction": _masked_mean((gray <= 0.01).float(), valid),
        "bright_saturation_fraction": _masked_mean((gray >= 0.99).float(), valid),
        "background_mean": bg_mean,
        "background_std": torch.sqrt(bg_var + 1e-12),
        "background_field_std": torch.sqrt(_masked_var(background, valid) + 1e-12),
        "background_high_frequency": _masked_mean(high_frequency, background_mask),
        "foreground_mean": fg_mean,
        "foreground_std": torch.sqrt(fg_var + 1e-12),
        "foreground_background_gap": gap,
        "cnr": cnr,
        "residual_p75": residual_q[:, 1],
        "residual_p90": residual_q[:, 2],
        "residual_p95": residual_q[:, 3],
        "residual_p99": residual_q[:, 4],
        "foreground_fraction": _masked_mean(foreground, valid),
        "laplacian_variance_foreground": _masked_var(laplace, foreground_dilated),
        "laplacian_variance_background": _masked_var(laplace, background_mask),
        "tenengrad_foreground": _masked_mean(gradient.square(), foreground_dilated),
        "tenengrad_background": _masked_mean(gradient.square(), background_mask),
        "brenner_x": _masked_mean(brenner_x, brenner_mask_x),
        "brenner_y": _masked_mean(brenner_y, brenner_mask_y),
        "high_frequency_foreground": _masked_mean(high_frequency, foreground_dilated),
        "high_frequency_background": _masked_mean(high_frequency, background_mask),
        "gradient_p90": gradient_q[:, 0],
        "gradient_p95": gradient_q[:, 1],
        "gradient_p99": gradient_q[:, 2],
        "strong_edge_fraction": _masked_mean((gradient > gradient_q[:, 1, None, None, None]).float(), valid),
        "weak_to_strong_edge_ratio": _masked_mean((gradient > gradient_q[:, 0, None, None, None]).float(), valid)
        / _masked_mean((gradient > gradient_q[:, 1, None, None, None]).float(), valid).clamp_min(1e-6),
        "valid_fraction": valid.flatten(1).mean(1),
    }
    scalar_cpu = {name: value.detach().cpu().numpy() for name, value in scalar.items()}
    mask_small = F.interpolate(foreground, size=(morphology_size, morphology_size), mode="nearest").detach().cpu().numpy()
    valid_small = F.interpolate(valid, size=(morphology_size, morphology_size), mode="nearest").detach().cpu().numpy()
    result: list[dict[str, float]] = []
    for index in range(gray.shape[0]):
        width, height = original_sizes[index]
        record = {name: float(values[index]) for name, values in scalar_cpu.items()}
        record.update(
            {
                "original_width": float(width),
                "original_height": float(height),
                "original_megapixels": float(width * height / 1_000_000.0),
                "original_aspect_ratio": float(width / max(height, 1)),
            }
        )
        record.update(morphology_features(mask_small[index, 0] > 0.5, valid_small[index, 0] > 0.5))
        result.append(record)
    return result


def extract_shard(
    manifest_path: Path,
    output_path: Path,
    device_name: str,
    shard_index: int,
    num_shards: int,
    image_size: int,
    batch_size: int,
    degradation_levels: int,
    include_benign: bool = True,
    max_images: int | None = None,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured two-GPU feature extraction")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    records = read_jsonl(manifest_path)
    records = [record for index, record in enumerate(records) if index % num_shards == shard_index]
    if max_images is not None:
        records = records[:max_images]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_records: list[dict] = []
    failures: list[dict] = []
    started = time.time()

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        loaded_records: list[dict] = []
        loaded_images: list[LoadedImage] = []
        for record in batch_records:
            try:
                loaded = load_image(Path(record["path"]), image_size)
            except Exception as exc:
                failures.append({"image_id": record["image_id"], "path": record["path"], "error": f"{type(exc).__name__}: {exc}"})
                continue
            loaded_records.append(record)
            loaded_images.append(loaded)
        if not loaded_images:
            continue
        gray = torch.stack([item.gray for item in loaded_images]).to(device, non_blocking=True)
        valid = torch.stack([item.valid for item in loaded_images]).to(device, non_blocking=True)
        sizes = [(item.width, item.height) for item in loaded_images]

        variants: list[tuple[str, int, float, torch.Tensor]] = [("original", 0, 0.0, gray)]
        if include_benign:
            variants.extend(
                [
                    ("benign_hflip", 0, 0.0, torch.flip(gray, dims=(-1,))),
                    ("benign_rot180", 0, 0.0, torch.flip(gray, dims=(-2, -1))),
                ]
            )
        for family in DEGRADATION_FAMILIES:
            for level in range(1, degradation_levels + 1):
                seed = stable_seed("quality_ranker_v1", shard_index, start, family, level)
                degraded = apply_degradation(gray, valid, family, level, seed)
                variants.append((family, level, level / degradation_levels, degraded))

        for family, level, severity, variant_tensor in variants:
            features = extract_batch_features(variant_tensor, valid, sizes)
            for record, feature_values in zip(loaded_records, features):
                identity = {
                    "image_id": record["image_id"],
                    "path": record["path"],
                    "relative_path": record["relative_path"],
                    "source": record["source"],
                    "case_id": record["case_id"],
                    "sha256": record["sha256"],
                    "near_duplicate_group": record["near_duplicate_group"],
                    "variant": family if family in {"original", "benign_hflip", "benign_rot180"} else f"{family}_l{level}",
                    "family": family,
                    "level": level,
                    "severity": severity,
                    "shard_index": shard_index,
                }
                identity.update(feature_values)
                output_records.append(identity)

        completed = min(start + batch_size, len(records))
        if completed % max(batch_size * 5, 1) == 0 or completed == len(records):
            elapsed = max(time.time() - started, 1e-6)
            speed = completed / elapsed
            eta = (len(records) - completed) / max(speed, 1e-6)
            print(
                f"shard={shard_index} device={device_name} images={completed}/{len(records)} "
                f"speed={speed:.2f}img/s eta={eta/60:.1f}min rows={len(output_records)}",
                flush=True,
            )

    frame = pd.DataFrame(output_records)
    frame.to_parquet(output_path, index=False)
    summary = {
        "status": "PASS" if not failures and len(frame) > 0 else "FAIL",
        "manifest": str(manifest_path.resolve()),
        "output": str(output_path.resolve()),
        "device": device_name,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "input_images": len(records),
        "output_rows": len(frame),
        "degradation_levels": degradation_levels,
        "variants_per_image": 1 + (len(BENIGN_VARIANTS) if include_benign else 0) + len(DEGRADATION_FAMILIES) * degradation_levels,
        "include_benign": include_benign,
        "feature_count": len([column for column in frame.columns if column not in IDENTITY_COLUMNS]),
        "failure_count": len(failures),
        "failures": failures,
        "elapsed_seconds": time.time() - started,
    }
    with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary
