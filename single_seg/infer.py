#!/usr/bin/env python3
"""Run packaged chromosome MaskDINO inference on one image or a directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model_code"))

from maskdino import add_maskdino_config  # noqa: E402
from preprocessing import prepare_image  # noqa: E402


EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
PALETTE = np.asarray(
    [
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
        [210, 245, 60], [0, 128, 128], [230, 190, 255], [128, 0, 0],
    ],
    dtype=np.uint8,
)


def build_cfg(config: Path, weights: Path, device: str):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(str(config))
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.DEVICE = device
    # MaskDINO reads class metadata while constructing the model.  This is an
    # in-memory inference-only catalog entry; no dataset or annotation is loaded.
    cfg.DATASETS.TRAIN = ("chromosome_inference_only",)
    cfg.DATASETS.TEST = ()
    MetadataCatalog.get("chromosome_inference_only").set(thing_classes=["chromosome"])
    cfg.freeze()
    return cfg


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]
    images = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not images:
        raise ValueError(f"No supported images under: {path}")
    return images


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Cannot encode output: {path}")
    encoded.tofile(path)


def render_overlay(image: np.ndarray, masks: np.ndarray) -> np.ndarray:
    output = image.copy()
    for index, mask in enumerate(masks):
        color = PALETTE[index % len(PALETTE)]
        output[mask] = (0.55 * output[mask] + 0.45 * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color.tolist(), 1)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Image file or directory")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/maskdino_R50_chromosome_inference.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/model_0059999.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args()

    images = collect_images(args.input.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg(args.config.resolve(), args.weights.resolve(), args.device)
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(str(args.weights.resolve()))

    summary = []
    for index, path in enumerate(images, start=1):
        prepared = prepare_image(path, cfg)
        inputs = {
            "image": prepared.model_tensor,
            "height": prepared.original_height,
            "width": prepared.original_width,
        }
        with torch.no_grad():
            prediction = model([inputs])[0]["instances"].to("cpu")
        keep = prediction.scores >= args.score_threshold
        prediction = prediction[keep]
        order = torch.argsort(prediction.scores, descending=True)
        prediction = prediction[order]
        masks = prediction.pred_masks.numpy().astype(bool)
        scores = prediction.scores.numpy()
        boxes = prediction.pred_boxes.tensor.numpy()

        stem = f"{index:04d}_{path.stem}"
        np.savez_compressed(args.output_dir / f"{stem}_instances.npz", masks=masks, scores=scores, boxes=boxes)
        label_map = np.zeros((prepared.original_height, prepared.original_width), dtype=np.uint16)
        for instance_id, mask in enumerate(masks, start=1):
            label_map[(label_map == 0) & mask] = instance_id
        write_png(args.output_dir / f"{stem}_instance_ids.png", label_map)
        write_png(args.output_dir / f"{stem}_overlay.png", render_overlay(prepared.original_bgr, masks))
        record = {
            "input_name": path.name,
            "height": prepared.original_height,
            "width": prepared.original_width,
            "score_threshold": args.score_threshold,
            "instance_count": int(len(masks)),
            "instances": [
                {"instance_id": i + 1, "score": float(score), "bbox_xyxy": box.tolist()}
                for i, (score, box) in enumerate(zip(scores, boxes))
            ],
        }
        (args.output_dir / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        summary.append(record)
        print(f"[{index}/{len(images)}] {path.name}: {len(masks)} instances", flush=True)

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
