# Road Damage Detection — Demo App

A self-contained Gradio app for showing the trained models live: upload an
image or video, pick a model, see annotated detections with confidence and
severity. Works fully offline once set up — no internet needed after
installing packages, no GPU required (CPU works, just slower).

## Setup (any laptop — Windows, Mac, Linux)

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install PyTorch (CPU-only build — much smaller download).
#    Skip this line and let requirements.txt pull the default build instead
#    if this laptop has an NVIDIA GPU and you want it used.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install everything else
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

This prints a local URL, usually `http://127.0.0.1:7860` — open it in a
browser. Pick a model from the dropdown (RF-DETR is the best-performing one
overall), upload a photo or video, click "Run detection".

## What's in `weights/`

All 6 models from the full comparison, ready to use:

| File | Model | Val mAP50 |
|---|---|---:|
| `RF-DETR.pth` | RF-DETR (**deployed / best**) | 91.2% |
| `YOLOv8m.pt` | YOLOv8m | 86.6% |
| `YOLO11m.pt` | YOLO11m | 85.7% |
| `YOLOv9c.pt` | YOLOv9c | 78.0% |
| `YOLOv10m.pt` | YOLOv10m | 49.8% |
| `RT-DETR-L.pt` | RT-DETR-L | 49.1% |

The dropdown only lists models whose weight file is actually present, so
you can trim `weights/` down to just `RF-DETR.pth` if you only want the
winner and a smaller folder to carry around.

## Notes

- First run of each model is slower (loading + warm-up); subsequent
  predictions with the same model are faster since it stays loaded in memory.
- Video processing is frame-by-frame — on a CPU-only laptop, a 20-second
  clip can take a minute or two. That's expected, not a bug.
- Severity (Low/Medium/High/Critical) is a rule-based heuristic based on
  detection size, class, confidence, and instance count — not a learned or
  validated classification, since no severity ground truth exists in the
  training data.
- This app duplicates the inference code from the training notebook on
  purpose (rather than importing it), so this folder runs completely on its
  own with nothing else needed from the rest of the project.
