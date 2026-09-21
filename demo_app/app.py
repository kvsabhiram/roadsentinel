"""Road damage detection demo (Gradio).

Standalone app -- does not import the training notebook. Reuses the exact
same inference logic (severity scoring, detection normalization) that was
built and tested there, copied here so this folder works on its own on a
laptop with just `pip install -r requirements.txt`.

Run:
    python app.py
Then open the printed local URL (usually http://127.0.0.1:7860).
"""
import json
from collections import Counter
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

WEIGHTS_DIR = Path(__file__).parent / "weights"
CLASS_NAMES = {0: "Pothole", 1: "Crack", 2: "Open Manhole"}

CONFIG = {
    "conf_threshold": 0.25,
    "iou_threshold": 0.7,
    "imgsz": 640,
}

# Same severity heuristic as the notebook (Section 15) -- rule-based, not a
# learned/validated classifier, since the dataset has no severity labels.
SEVERITY_CONFIG = {
    "area_weight": {"Pothole": 1.0, "Crack": 0.6, "Open Manhole": 1.4},
    "area_thresholds": {"low": 0.01, "medium": 0.04, "high": 0.10},
    "multi_instance_bump_at": 3,
    "min_confidence_for_critical": 0.70,
}
SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]

# Model registry: label shown in the UI -> (weights filename, family, val mAP50).
# Only entries whose weights file actually exists under weights/ are offered.
MODEL_REGISTRY = {
    "RF-DETR (best, val mAP50 91.2%)": ("RF-DETR.pth", "rfdetr", 0.9118),
    "YOLOv8m (val mAP50 86.6%)": ("YOLOv8m.pt", "yolo", 0.8663),
    "YOLO11m (val mAP50 85.7%)": ("YOLO11m.pt", "yolo", 0.8575),
    "YOLOv9c (val mAP50 78.0%)": ("YOLOv9c.pt", "yolo", 0.7798),
    "YOLOv10m (val mAP50 49.8%)": ("YOLOv10m.pt", "yolo", 0.4985),
    "RT-DETR-L (val mAP50 49.1%)": ("RT-DETR-L.pt", "rtdetr", 0.4912),
}

_loaded_models = {}  # label -> loaded model object, populated lazily


def available_models():
    return [label for label, (fname, _, _) in MODEL_REGISTRY.items() if (WEIGHTS_DIR / fname).exists()]


def get_model(label):
    if label in _loaded_models:
        return _loaded_models[label]
    fname, family, _ = MODEL_REGISTRY[label]
    weights_path = WEIGHTS_DIR / fname
    if family == "rfdetr":
        from rfdetr import from_checkpoint
        model = from_checkpoint(str(weights_path))
    else:
        from ultralytics import RTDETR, YOLO
        model_class = RTDETR if family == "rtdetr" else YOLO
        model = model_class(str(weights_path))
    _loaded_models[label] = model
    return model, family


def severity_for_detection(class_name, confidence, relative_area, instance_count):
    weighted_area = relative_area * SEVERITY_CONFIG["area_weight"].get(class_name, 1.0)
    thresholds = SEVERITY_CONFIG["area_thresholds"]
    if weighted_area < thresholds["low"]:
        level_index = 0
    elif weighted_area < thresholds["medium"]:
        level_index = 1
    elif weighted_area < thresholds["high"]:
        level_index = 2
    else:
        level_index = 3
    if instance_count >= SEVERITY_CONFIG["multi_instance_bump_at"]:
        level_index = min(level_index + 1, len(SEVERITY_LEVELS) - 1)
    if level_index == len(SEVERITY_LEVELS) - 1 and confidence < SEVERITY_CONFIG["min_confidence_for_critical"]:
        level_index -= 1
    return SEVERITY_LEVELS[level_index]


def normalize_detections(prediction, model_family):
    if model_family == "rfdetr":
        return [
            {"class_id": int(class_id), "confidence": float(confidence), "xyxy": tuple(float(v) for v in box)}
            for box, confidence, class_id in zip(prediction.xyxy, prediction.confidence, prediction.class_id)
        ]
    return [
        {
            "class_id": int(box.cls.item()),
            "confidence": float(box.conf.item()),
            "xyxy": tuple(float(v) for v in box.xyxy[0].tolist()),
        }
        for box in prediction.boxes
    ]


def score_and_annotate(image_rgb, raw_detections):
    image_height, image_width = image_rgb.shape[:2]
    class_counts = Counter(CLASS_NAMES[d["class_id"]] for d in raw_detections)
    annotated = image_rgb.copy()
    detections = []
    for det in raw_detections:
        class_name = CLASS_NAMES[det["class_id"]]
        x1, y1, x2, y2 = det["xyxy"]
        relative_area = ((x2 - x1) * (y2 - y1)) / (image_width * image_height)
        severity = severity_for_detection(class_name, det["confidence"], relative_area, class_counts[class_name])
        label = f"{class_name} {det['confidence'] * 100:.1f}% [{severity}]"
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        (_, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(int(y1) - 6, text_h)
        cv2.putText(annotated, label, (int(x1), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        detections.append({
            "class": class_name,
            "confidence": round(det["confidence"], 4),
            "severity": severity,
        })
    return annotated, detections


def run_image_inference(model, model_family, image_rgb):
    conf = CONFIG["conf_threshold"]
    if model_family == "rfdetr":
        prediction = model.predict(image_rgb, threshold=conf)
    else:
        prediction = model.predict(
            source=image_rgb, imgsz=CONFIG["imgsz"], conf=conf, iou=CONFIG["iou_threshold"], verbose=False
        )[0]
    raw_detections = normalize_detections(prediction, model_family)
    return score_and_annotate(image_rgb, raw_detections)


def run_video_inference(model, model_family, video_path, output_path):
    conf = CONFIG["conf_threshold"]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    frame_count, detection_count = 0, 0
    try:
        while True:
            read_ok, frame_bgr = capture.read()
            if not read_ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if model_family == "rfdetr":
                prediction = model.predict(frame_rgb, threshold=conf)
            else:
                prediction = model.predict(
                    source=frame_rgb, imgsz=CONFIG["imgsz"], conf=conf, iou=CONFIG["iou_threshold"], verbose=False
                )[0]
            raw_detections = normalize_detections(prediction, model_family)
            annotated_rgb, _ = score_and_annotate(frame_rgb, raw_detections)
            writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
            frame_count += 1
            detection_count += len(raw_detections)
    finally:
        capture.release()
        writer.release()
    return frame_count, detection_count


def predict_image(model_label, image):
    if image is None:
        return None, "Upload an image first."
    if not model_label:
        return None, "Select a model first."
    model, family = get_model(model_label)
    annotated_rgb, detections = run_image_inference(model, family, image)
    if not detections:
        summary = "No detections above the confidence threshold (0.25)."
    else:
        summary = "\n".join(f"{d['class']} — Confidence: {d['confidence'] * 100:.1f}% — Severity: {d['severity']}" for d in detections)
    return annotated_rgb, summary


def predict_video(model_label, video):
    if video is None:
        return None, "Upload a video first."
    if not model_label:
        return None, "Select a model first."
    model, family = get_model(model_label)
    output_path = Path(video).with_name(Path(video).stem + "_predicted.mp4")
    frame_count, detection_count = run_video_inference(model, family, video, output_path)
    summary = f"Processed {frame_count} frames, {detection_count} total detections."
    return str(output_path), summary


with gr.Blocks(title="Road Damage Detection") as demo:
    gr.Markdown(
        "# Road Damage Detection\n"
        "Detects potholes, cracks, and open manholes. Pick a model (RF-DETR is the best-performing "
        "one from the full comparison), upload an image or video, and see annotated detections "
        "with confidence and a rule-based severity estimate."
    )
    model_dropdown = gr.Dropdown(
        choices=available_models(),
        value=available_models()[0] if available_models() else None,
        label="Model",
    )
    with gr.Tab("Image"):
        with gr.Row():
            image_input = gr.Image(label="Input image", type="numpy")
            image_output = gr.Image(label="Annotated result", type="numpy")
        image_summary = gr.Textbox(label="Detections", lines=6)
        image_button = gr.Button("Run detection", variant="primary")
        image_button.click(predict_image, inputs=[model_dropdown, image_input], outputs=[image_output, image_summary])
    with gr.Tab("Video"):
        with gr.Row():
            video_input = gr.Video(label="Input video")
            video_output = gr.Video(label="Annotated result")
        video_summary = gr.Textbox(label="Summary", lines=2)
        video_button = gr.Button("Run detection", variant="primary")
        video_button.click(predict_video, inputs=[model_dropdown, video_input], outputs=[video_output, video_summary])

if __name__ == "__main__":
    if not available_models():
        print(
            f"No model weights found in {WEIGHTS_DIR}/. "
            "Copy at least RF-DETR.pth there (see README.md) before running."
        )
    demo.launch()
