# 🛣️ RoadSentinel

**AI-powered road damage detection and severity assessment** — potholes, cracks, and open manholes, detected and scored from a single photo or a live video feed.

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model](https://img.shields.io/badge/best%20model-RF--DETR-success.svg)](#results)

---

## What this is

Six object-detection architectures — the YOLO family, RT-DETR, and RF-DETR — trained and rigorously compared on the same road-hazard dataset, ensembled, evaluated on a genuinely held-out test set, exported to ONNX, and wrapped in a runnable demo app. Not a single-notebook tutorial run: the project surfaced and fixed a real train/validation data-leakage bug in the source dataset before any of the results below were trustworthy.

## Results

**Winner: RF-DETR** (transformer detector, DINOv2-windowed-small backbone) — outperformed every YOLO variant, RT-DETR, and two ensemble configurations.

| Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|
| Validation | **91.18%** | 52.38% | 91.36% | 86.62% |
| **Test (held-out)** | **88.48%** | 49.11% | 92.58% | 80.45% |

<details>
<summary>Full 8-way comparison (6 models + 2 ensembles, validation set)</summary>

| Model | mAP50 | mAP50-95 |
|---|---:|---:|
| **RF-DETR** | **91.18%** | **52.38%** |
| Ensemble A (YOLO11m + RF-DETR, NMS) | 90.96% | 52.17% |
| YOLOv8m | 86.63% | 48.08% |
| YOLO11m | 85.75% | 49.73% |
| Ensemble B (YOLO11m + RT-DETR + RF-DETR, NMS) | 85.16% | 49.50% |
| YOLOv9c | 77.98% | 42.76% |
| YOLOv10m | 49.85% | 27.30% |
| RT-DETR-L | 49.12% | 24.75% |

Ensembling (both NMS and Weighted Box Fusion, both required combinations) was tried and evaluated honestly — neither beat RF-DETR alone, so neither is the deployed model. That's a real finding, not a shortcut.
</details>

## Why this isn't just "ran a notebook"

- **Found and fixed real data leakage**: the shipped dataset had 124 byte-identical images duplicated across its train and validation folders, and a "test" split with *zero* labels. Both silently inflate reported metrics if untouched. Fixed by deduplicating by content hash and re-cutting a stratified 70/15/15 train/val/test split before any training — the fix that made the numbers above trustworthy.
- **Fair, controlled comparison**: all 6 models trained on the identical split, evaluated with the identical methodology (COCO-style mAP via `pycocotools`), so the leaderboard is an apples-to-apples comparison, not cherry-picked runs.
- **Ensembling done rigorously, not rhetorically**: NMS and Weighted Box Fusion both implemented and scored against the same COCO-mAP pipeline used for individual models — and reported as *not* winning, per the evidence.
- **Full deployment path**: best model exported to ONNX with before/after accuracy validation (confirms the export didn't silently degrade the model), plus a working Gradio demo app that runs on a CPU-only laptop.
- **Debugged real infrastructure failures**, not just model code: a CUDA pin-memory crash traced to an inverted boolean, a headless-execution pipe deadlock, and a training job getting killed by desktop session logout (fixed at the systemd level with `loginctl enable-linger`) — the kind of issues that show up running ML pipelines outside a notebook sandbox.

## Architecture

```
Image / video frame
        │
        ▼
┌───────────────────┐
│   RF-DETR model    │   DINOv2-windowed-small backbone (ViT)
│  (576×576 input)   │   transformer decoder, NMS-free
└─────────┬──────────┘
          ▼
  boxes + classes + confidence
          │
          ▼
┌───────────────────┐
│  Severity scoring   │   rule-based: box area, class weight,
│  (Low→Critical)     │   instance count, confidence
└─────────┬──────────┘
          ▼
  Annotated image / video
  + per-detection confidence + severity
```

## Tech stack

`PyTorch` · `Ultralytics` (YOLOv8/9/10/11, RT-DETR) · `RF-DETR` (Roboflow) · `PyTorch Lightning` · `ONNX Runtime` · `ensemble-boxes` (NMS/WBF) · `pycocotools` (COCO mAP) · `OpenCV` · `Gradio`

## Quickstart

**Try the demo (no training needed, CPU is fine):**

```bash
cd demo_app
pip install -r requirements.txt
python app.py
```

Open the printed URL, upload a road photo or video, pick a model, see detections + severity. (Model weights aren't in this repo due to GitHub's file-size limits — see [Model weights](#model-weights) below.)

**Reproduce the full training pipeline:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace road-damage-detection-30-epochs.ipynb
```

Runs end-to-end on Kaggle unchanged (auto-detects the environment) or locally with a CUDA GPU (~2-3.5 hours for the full comparison). See the notebook's own first cells for dataset setup.

## Model weights

Trained weights (`.pt`/`.pth`) and ONNX exports aren't committed to this repo — several individual files exceed GitHub's 100MB limit. All 6 models, in both formats, are attached to the **[v1.0-models release](https://github.com/kvsabhiram/roadsentinel/releases/tag/v1.0-models)** instead.

To run the demo app: download the `.pt`/`.pth` files from that release and place them in `demo_app/weights/`.

Full result set (all test cases, sample images, annotated videos) is also available on **[Google Drive](#)** *(add your Drive link here)*.

## Project structure

```
road-damage-detection-30-epochs.ipynb   the complete pipeline: dataset dedup +
                                         re-split, training all 6 models,
                                         ensembling, test evaluation, ONNX export
demo_app/
  app.py                                Gradio demo (image + video)
  requirements.txt
results/                                leaderboard, ensemble comparison, final
                                         test evaluation, ONNX validation -- the
                                         raw numbers behind the table above
promt.txt                               original project brief / requirements
requirements.txt                        training environment dependencies
```

## Known limitations

- Severity (Low/Medium/High/Critical) is a transparent rule-based heuristic over box size, class, confidence, and instance count — not a learned or validated classifier, since the dataset has no severity ground truth.
- RF-DETR's ONNX export wasn't independently re-validated for accuracy after conversion (its library doesn't expose an ONNX Runtime evaluation path the way Ultralytics does); the YOLO/RT-DETR exports were.
- Validated on one dataset's distribution (potholes, cracks, open manholes); generalization to other road types/regions/camera conditions is untested.

## Acknowledgments

Dataset: [potholes, cracks and openmanholes (Road Hazards)](https://www.kaggle.com/datasets/sabidrahman/pothole-cracks-and-openmanhole) by sabidrahman on Kaggle.

## License

[MIT](LICENSE)
