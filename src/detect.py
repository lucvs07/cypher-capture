"""
Real-time PPE detection from webcam, video file, or image directory.

Usage (run from the project root):
    python src/detect.py --model models/best.pt --source usb0
    python src/detect.py --model models/best.pt --source video.mp4 --thresh 0.45
    python src/detect.py --model models/best.pt --source images/ --no-upload
"""
# Imports
import argparse
import os
import sys
import glob
import time
import requests
from pathlib import Path

# Ensure project root is on sys.path regardless of how this script is invoked
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

from src.utils.visualization import draw_detections, VIOLATION_CLASSES
from src.utils.upload import build_snapshot, upload_snapshot

load_dotenv()

OUTPUTS_DIR = Path("outputs/captures")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------------

def open_source(source: str):
    if source.startswith("usb"):
        idx = int(source.replace("usb", ""))
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            sys.exit(f"Cannot open camera index {idx}.")
        return cap, "camera"

    if os.path.isfile(source):
        ext = Path(source).suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return source, "image"
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            sys.exit(f"Cannot open video file: {source}")
        return cap, "video"

    if os.path.isdir(source):
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(source, p)))
        if not files:
            sys.exit(f"No images found in directory: {source}")
        return sorted(files), "image_dir"

    sys.exit(f"Unrecognised source: {source}")


# ---------------------------------------------------------------------------
# Violation check
# ---------------------------------------------------------------------------

def has_violation(result, conf_threshold: float) -> bool:
    names = result.names
    for box in result.boxes:
        if float(box.conf) >= conf_threshold and names[int(box.cls)] in VIOLATION_CLASSES:
            return True
    return False


# ---------------------------------------------------------------------------
# Upload and notify
# ---------------------------------------------------------------------------

def handle_upload(frame_raw, frame_annotated, track_id: int, endpoint: str | None):
    snapshot = build_snapshot(track_id, frame_raw, frame_annotated)
    try:
        urls = upload_snapshot(snapshot)
        print(f"  [upload] raw={urls['raw']}")
        print(f"  [upload] annotated={urls['annotated']}")
        if endpoint:
            payload = {"snapshot_id": snapshot.uid, **urls}
            requests.post(endpoint, json=payload, timeout=5)
    except Exception as exc:
        print(f"  [upload error] {exc}")


# ---------------------------------------------------------------------------
# Inference loops
# ---------------------------------------------------------------------------

def run_stream(cap, model: YOLO, args):
    frame_id = 0
    upload_cooldown = {}

    print("Streaming. Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated = draw_detections(frame, results[0], args.thresh)

        frame_id += 1
        if has_violation(results[0], args.thresh):
            now = time.time()
            last = upload_cooldown.get(0, 0)
            if now - last > args.cooldown and not args.no_upload:
                upload_cooldown[0] = now
                handle_upload(frame, annotated, frame_id, os.getenv("API_ENDPOINT"))

        cv2.imshow("Cypher Capture — PPE Detection (Q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_images(sources, model: YOLO, args):
    for path in sources:
        frame = cv2.imread(path)
        if frame is None:
            print(f"Skipping unreadable file: {path}")
            continue

        results = model(frame, verbose=False)
        annotated = draw_detections(frame, results[0], args.thresh)

        out_path = OUTPUTS_DIR / f"result_{Path(path).stem}.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved: {out_path}")

        if has_violation(results[0], args.thresh) and not args.no_upload:
            handle_upload(frame, annotated, 0, os.getenv("API_ENDPOINT"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Cypher Capture — PPE Detection")
    parser.add_argument("--model", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--source", required=True,
                        help="usb0 | path/to/video.mp4 | path/to/image.jpg | path/to/dir/")
    parser.add_argument("--thresh", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--cooldown", type=float, default=5.0,
                        help="Seconds between uploads for live streams (default: 5)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Disable Cloudinary upload")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    source, kind = open_source(args.source)

    if kind in {"camera", "video"}:
        run_stream(source, model, args)
    elif kind == "image":
        run_images([source], model, args)
    elif kind == "image_dir":
        run_images(source, model, args)


if __name__ == "__main__":
    main()
