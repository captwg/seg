# Traditional Quality Ranker V4 - frozen runtime

This is the minimal runnable release of the current V4 chromosome-image quality
ranker. It contains only the frozen calibration, inference-time feature code,
the scoring command, dependency list, and integrity manifest.

It intentionally excludes training and calibration datasets, test images,
enhancement outputs, reports, and enhancement code.

## Interpretation boundary

The result is a no-GT traditional image-quality proxy. It is not Dice, AP, a
segmentation-success probability, or evidence that every chromosome was
preserved after enhancement.

The score is a weighted sum of seven fixed-ECDF component scores, each
calibrated against 4,455 frozen training-reference images:

| Component | Weight |
| --- | ---: |
| Segmentation proxy | 40% |
| Individual clarity | 22% |
| Morphology and distribution | 14% |
| Gray visibility | 10% |
| Focus | 6% |
| Background illumination | 6% |
| Original acquisition specification | 2% |

The segmentation proxy uses a traditional connected-component estimate relative
to 46 chromosomes; it is not ground truth.

## Install and run

Use Python 3.10+ in a clean environment:

    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt
    python scripts/score_directory.py /path/to/images /path/to/scores.csv --workers 8

The input directory is searched recursively for JPG, JPEG, PNG, TIFF, and BMP
images. The CSV contains a cross-batch-comparable score (0-100), estimated
chromosome count, all seven component scores/contributions, and input hashes.
rank_in_batch is only the ordering within this invocation.

## Integrity and identity

- Model: models/hierarchical_46_fixed_ranker.joblib
- Model SHA256:
  49e4c31a4614d08efcde6ccbb4ef5eecd76062daf6f8690ec8243ff19a21e04c
- Version: traditional_quality_ranker_v4_hierarchical_46
- Verify all distributed files: sha256sum -c MANIFEST.sha256

The command is runtime-only: it does not train, recalibrate, modify inputs, or
perform image enhancement.
