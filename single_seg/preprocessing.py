#!/usr/bin/env python3
"""Exact deterministic inference preprocessing for the packaged MaskDINO model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2.data import transforms as T


@dataclass(frozen=True)
class PreparedImage:
    original_bgr: np.ndarray
    model_tensor: torch.Tensor
    original_height: int
    original_width: int


def read_bgr(path: Path) -> np.ndarray:
    """Read paths containing non-ASCII characters without changing bit depth."""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def prepare_image(path: Path, cfg) -> PreparedImage:
    """Match Detectron2 DefaultPredictor: RGB conversion and shortest-edge resize."""
    original_bgr = read_bgr(path)
    height, width = original_bgr.shape[:2]
    image = original_bgr
    if cfg.INPUT.FORMAT == "RGB":
        image = image[:, :, ::-1]
    elif cfg.INPUT.FORMAT != "BGR":
        raise ValueError(f"Unsupported INPUT.FORMAT={cfg.INPUT.FORMAT!r}")

    augmentation = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST,
    )
    resized = augmentation.get_transform(image).apply_image(image)
    tensor = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1))
    return PreparedImage(original_bgr, tensor, height, width)
