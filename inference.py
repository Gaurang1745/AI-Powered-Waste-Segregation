"""
Real-time webcam and static image waste classification.

Usage:
    python inference.py --mode webcam
    python inference.py --mode image --source path/to/image.jpg
    python inference.py --mode image --source path/to/folder/
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO
from utils import load_config


def get_display_info(class_name, config):
    """Get display label and BGR color for a class."""
    class_cfg = config["classes"].get(class_name, {})
    label = class_cfg.get("label", class_name)
    color = tuple(class_cfg.get("color", [255, 255, 255]))
    return label, color


def draw_classification(frame, class_name, confidence, config):
    """Draw classification result overlay on frame."""
    label, color = get_display_info(class_name, config)
    text = f"{label}: {confidence:.1%}"

    # Semi-transparent background bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Classification text
    cv2.putText(
        frame, text, (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA,
    )

    # Color indicator bar
    cv2.rectangle(frame, (0, 55), (frame.shape[1], 60), color, -1)

    return frame


def draw_uncertain(frame):
    """Draw uncertain label when confidence is below threshold."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(
        frame, "Uncertain", (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 128, 128), 2, cv2.LINE_AA,
    )
    return frame


def classify_frame(model, frame, config):
    """Run classification on a single frame. Returns annotated frame, class name, confidence."""
    inf_cfg = config["inference"]
    results = model(frame, imgsz=inf_cfg["imgsz"], verbose=False)

    probs = results[0].probs
    top1_idx = probs.top1
    top1_conf = float(probs.top1conf)
    class_name = results[0].names[top1_idx]

    if top1_conf >= inf_cfg["confidence_threshold"]:
        frame = draw_classification(frame, class_name, top1_conf, config)
    else:
        frame = draw_uncertain(frame)

    return frame, class_name, top1_conf


def run_webcam(model, config):
    """Real-time webcam classification loop."""
    inf_cfg = config["inference"]
    cap = cv2.VideoCapture(inf_cfg["webcam_source"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, inf_cfg["display_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, inf_cfg["display_height"])

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam started. Press 'q' to quit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, class_name, confidence = classify_frame(model, frame, config)

        # FPS counter
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        cv2.imshow("Waste Segregation - YOLOv8", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(model, source, config):
    """Classify a single image or all images in a folder."""
    source_path = Path(source)

    if source_path.is_dir():
        image_files = sorted(
            p for p in source_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not image_files:
            print(f"No images found in {source_path}")
            return
    else:
        if not source_path.exists():
            print(f"File not found: {source_path}")
            return
        image_files = [source_path]

    print(f"Classifying {len(image_files)} image(s)...\n")

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Could not read: {img_path}")
            continue

        frame, class_name, confidence = classify_frame(model, frame, config)

        label, _ = get_display_info(class_name, config)
        print(f"  {img_path.name}  ->  {label} ({confidence:.1%})")

        cv2.imshow("Waste Segregation - YOLOv8", frame)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Waste classification inference")
    parser.add_argument(
        "--mode", choices=["webcam", "image"], required=True,
        help="Inference mode: webcam for real-time, image for static files",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Image path or folder (required for --mode image)",
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model path from config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model or config["inference"]["model_path"]

    if not Path(model_path).exists():
        print(f"Model not found at {model_path}.")
        print("Train the model first with: python train.py")
        return

    model = YOLO(model_path)

    if args.mode == "webcam":
        run_webcam(model, config)
    elif args.mode == "image":
        if not args.source:
            print("--source is required for image mode.")
            return
        run_image(model, args.source, config)


if __name__ == "__main__":
    main()
