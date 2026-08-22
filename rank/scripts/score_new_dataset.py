#!/usr/bin/env python3
"""Score an image directory with the frozen V4 hierarchical ranker."""
from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

from build_and_score import IDENTITY, WEIGHTS, object_feature_worker, score_frame
from traditional_quality_ranker.features import extract_batch_features, load_image


EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    paths = sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not paths:
        raise RuntimeError(f"No supported images under {input_dir}")

    base_rows: list[dict] = []
    for start in range(0, len(paths), 16):
        batch_paths = paths[start:start + 16]
        loaded = [load_image(path, 384) for path in batch_paths]
        gray = torch.stack([item.gray for item in loaded])
        valid = torch.stack([item.valid for item in loaded])
        features = extract_batch_features(gray, valid, [(item.width, item.height) for item in loaded])
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
    base = pd.DataFrame(base_rows)

    inputs = [(row["sha256"], row["path"]) for row in base_rows]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        object_rows = list(executor.map(object_feature_worker, inputs, chunksize=4))
    frame = base.merge(pd.DataFrame(object_rows), on="sha256", validate="one_to_one")
    bundle = joblib.load(ROOT / "models" / "hierarchical_46_fixed_ranker.joblib")
    scored = score_frame(frame, bundle["references"])
    scored = scored.sort_values(["score", "sha256"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    scored.insert(0, "rank_in_batch", np.arange(1, len(scored) + 1))
    detail = ["estimated_chromosome_count", "count_absolute_error", "single_object_gray_range_median", "single_object_gray_std_median"]
    category = [column for name in WEIGHTS for column in (f"{name}_score", f"{name}_contribution")]
    output = scored[["rank_in_batch", "score"] + detail + category + IDENTITY]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(args.output_csv.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
