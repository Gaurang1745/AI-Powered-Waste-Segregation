"""
Fine-tune YOLOv8 classification model for waste segregation.

Usage:
    python train.py                          # Train with config.yaml defaults
    python train.py --epochs 100             # Override specific parameters
    python train.py --resume                 # Resume interrupted training
"""

import argparse
from pathlib import Path

from ultralytics import YOLO
from utils import load_config


def parse_args():
    """Parse CLI arguments that can override config values."""
    parser = argparse.ArgumentParser(description="Train waste classification model")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def train(config, args):
    """Execute training pipeline."""
    cfg = config["training"]

    # CLI args override config
    epochs = args.epochs or cfg["epochs"]
    batch = args.batch or cfg["batch"]
    imgsz = args.imgsz or cfg["imgsz"]
    device = args.device if args.device is not None else cfg["device"]

    # Verify dataset exists
    data_path = Path(cfg["data"])
    if not data_path.exists():
        print(f"Dataset not found at {data_path}. Run prepare_dataset.py first.")
        return
    if not (data_path / "train").exists():
        print(f"Missing train/ directory in {data_path}. Run prepare_dataset.py first.")
        return

    # Load model
    if args.resume:
        model_path = Path(cfg["project"]) / cfg["name"] / "weights" / "last.pt"
        if not model_path.exists():
            print(f"No checkpoint found at {model_path} for resuming.")
            return
        print(f"Resuming training from {model_path}")
        model = YOLO(str(model_path))
    else:
        model_name = args.model or cfg["model"]
        print(f"Loading pre-trained model: {model_name}")
        model = YOLO(model_name)

    # Train
    print(f"\nTraining configuration:")
    print(f"  Data:      {data_path.resolve()}")
    print(f"  Epochs:    {epochs}")
    print(f"  Batch:     {batch}")
    print(f"  Image size:{imgsz}")
    print(f"  Optimizer: {cfg['optimizer']}")
    print(f"  LR:        {cfg['lr0']}")
    print(f"  Device:    {device or 'auto'}")
    print()

    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=cfg["patience"],
        lr0=cfg["lr0"],
        optimizer=cfg["optimizer"],
        project=cfg["project"],
        name=cfg["name"],
        pretrained=cfg["pretrained"],
        device=device if device else None,
        exist_ok=True,
    )

    # Evaluate on test set
    best_path = Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"
    if best_path.exists():
        print(f"\nEvaluating best model on test set...")
        best_model = YOLO(str(best_path))
        metrics = best_model.val(data=str(data_path), split="test", imgsz=imgsz)

        print(f"\n{'='*50}")
        print(f"  Test Top-1 Accuracy: {metrics.top1:.4f}")
        print(f"  Test Top-5 Accuracy: {metrics.top5:.4f}")
        print(f"{'='*50}")
        print(f"\nBest model saved to: {best_path.resolve()}")
        print(f"Run inference with: python inference.py --mode webcam")
    else:
        print("Warning: best.pt not found after training.")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config, args)
