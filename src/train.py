"""
Train a YOLO model on the Construction-PPE dataset.

Usage:
    python src/train.py
    python src/train.py --model yolo11n.pt --epochs 50 --imgsz 640 --batch 16
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

CONFIG = Path("config/construction-ppe.yaml")
OUTPUT_DIR = Path("outputs/runs")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPE detector with Ultralytics YOLO")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Base model to fine-tune (default: yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early-stopping patience in epochs")
    parser.add_argument("--device", default=None,
                        help="Training device: 0 | cpu | mps (auto-detected if omitted)")
    parser.add_argument("--name", default="ppe_train",
                        help="Run name under outputs/runs/")
    return parser.parse_args()


def main():
    args = parse_args()

    if not CONFIG.exists():
        raise FileNotFoundError(
            f"Dataset config not found at {CONFIG}. "
            "Run `python scripts/download_dataset.py` first."
        )

    model = YOLO(args.model)

    model.train(
        data=str(CONFIG),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(OUTPUT_DIR),
        name=args.name,
        exist_ok=True,
    )

    best = OUTPUT_DIR / args.name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights saved to: {best}")
    print(f"Copy to models/ to use with detect.py:  cp {best} models/best.pt")


if __name__ == "__main__":
    main()
