#!/usr/bin/env python3
"""Score all supported images under a directory with frozen V4."""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quality_ranker.features import extract_batch_features, load_image
from quality_ranker.v4_runtime import (
    IDENTITY,
    WEIGHTS,
    load_ranker,
    object_feature_worker,
    score_frame,
    sha256_file,
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory searched recursively for images")
    parser.add_argument("output_csv", type=Path, help="CSV file to create")
    parser.add_argument("--workers", type=int, default=8, help="Object-feature worker processes")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "hierarchical_46_fixed_ranker.joblib",
        help="Frozen V4 joblib model",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    input_dir = args.input_dir.resolve()
    paths = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No supported images under {input_dir}")

    base_rows: list[dict] = []
    for start in range(0, len(paths), 16):
        batch_paths = paths[start:start + 16]
        loaded = [load_image(path, 384) for path in batch_paths]
        features = extract_batch_features(
            torch.stack([item.gray for item in loaded]),
            torch.stack([item.valid for item in loaded]),
            [(item.width, item.height) for item in loaded],
        )
        for path, values in zip(batch_paths, features):
            sha = sha256_file(path)
            values.update({
                "image_id": sha[:16],
                "relative_path": str(path.relative_to(input_dir)),
                "path": str(path),
                "source": input_dir.name,
                "case_id": input_dir.name,
                "sha256": sha,
                "near_duplicate_group": sha,
            })
            base_rows.append(values)
        print(f"base_features {min(start + 16, len(paths))}/{len(paths)}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        object_rows = list(
            executor.map(
                object_feature_worker,
                [(row["sha256"], row["path"]) for row in base_rows],
                chunksize=4,
            )
        )
    frame = pd.DataFrame(base_rows).merge(
        pd.DataFrame(object_rows), on="sha256", validate="one_to_one"
    )
    bundle = load_ranker(args.model.resolve())
    scored = score_frame(frame, bundle["references"])
    scored = scored.sort_values(
        ["score", "sha256"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    scored.insert(0, "rank_in_batch", np.arange(1, len(scored) + 1))
    detail = [
        "estimated_chromosome_count",
        "count_absolute_error",
        "single_object_gray_range_median",
        "single_object_gray_std_median",
    ]
    category = [
        column for name in WEIGHTS
        for column in (f"{name}_score", f"{name}_contribution")
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored[["rank_in_batch", "score"] + detail + category + IDENTITY].to_csv(
        args.output_csv, index=False, encoding="utf-8"
    )
    print(args.output_csv.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
