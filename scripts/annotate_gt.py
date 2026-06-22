#!/usr/bin/env python3
"""
Ground-truth annotation tool for posture evaluation (Phase 2, Day 1).

Two phases in a single run:
  A. Interactive labelling (fast, NO model) — play the clip and press a key to
     set the posture label that applies from the current frame onward.
  B. Automatic expansion (runs the pose model) — re-reads the clip and emits one
     per-frame annotation (bbox + keypoints + your label) in the EXACT schema
     scripts/evaluate_pose.py expects.

IMPORTANT — PCK caveat:
  The GT keypoints written in phase B come from the SAME detector that
  evaluate_pose.py runs, so PCK measured against this file is self-referential
  (~1.0) and is NOT a valid keypoint-accuracy metric. This tool exists to
  measure the CLASSIFICATION metrics — recall (immediate_risk), accuracy and
  false-alarm rate — where your key presses are the independent ground truth.
  A real PCK needs manually-labelled keypoints (e.g. CVAT). See
  docs/plano-testes-dia1.md §2c.

Label keys:  n = normal   e = ergonomically_inadequate
             i = immediate_risk   z = critical_zone
Controls:    SPACE = pause/play   . = step one frame (paused)
             + / - = faster / slower   u = undo last label change   q = finish & save

Usage:
    python scripts/annotate_gt.py \
        --video data/validation/cenario1.mp4 \
        --output data/annotations/cenario1_gt.json \
        [--pose-model yolo11s-pose.pt] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.video import open_capture

LABEL_KEYS = {
    ord("n"): "normal",
    ord("e"): "ergonomically_inadequate",
    ord("i"): "immediate_risk",
    ord("z"): "critical_zone",
}
COLOR = {
    "normal": (0, 200, 0),
    "ergonomically_inadequate": (0, 180, 220),
    "immediate_risk": (0, 0, 230),
    "critical_zone": (0, 90, 230),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pose-model", default="yolo11s-pose.pt")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def phase_a_label(video: str) -> dict[int, str]:
    """Interactive labelling. Returns {frame_id: label} for every frame seen."""
    cap = open_capture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    changes: list[tuple[int, str]] = [(0, "normal")]  # ordered (frame_id, label) change points
    fid = 0
    paused = False
    delay = 30                  # ms between frames; +/- adjusts
    cur = "normal"
    last_frame = None

    print("\nPhase A — labelling.")
    print("  n/e/i/z set label | SPACE pause | . step | +/- speed | u undo | q finish\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame
            fid += 1

        canvas = last_frame.copy() if last_frame is not None else np.zeros((480, 640, 3), np.uint8)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
        txt = f"f{fid}/{total}  label={cur}  {'PAUSED' if paused else 'play'}  delay={delay}ms"
        cv2.putText(canvas, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    COLOR.get(cur, (255, 255, 255)), 2)
        cv2.imshow("annotate_gt — Phase A", canvas)

        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord(".") and paused:
            ret, f2 = cap.read()
            if ret:
                last_frame = f2
                fid += 1
        elif key in (ord("+"), ord("=")):
            delay = max(1, delay - 10)
        elif key == ord("-"):
            delay = min(200, delay + 10)
        elif key == ord("u") and len(changes) > 1:
            changes.pop()
            cur = changes[-1][1]
            print(f"  undo -> label now {cur}")
        elif key in LABEL_KEYS:
            cur = LABEL_KEYS[key]
            changes.append((fid, cur))
            print(f"  f{fid}: -> {cur}")

    cap.release()
    cv2.destroyAllWindows()

    # Expand change points into a per-frame label map
    per_frame: dict[int, str] = {}
    ci = 0
    label = changes[0][1]
    for f in range(fid + 1):
        while ci + 1 < len(changes) and changes[ci + 1][0] <= f:
            ci += 1
            label = changes[ci][1]
        per_frame[f] = label
    return per_frame


def phase_b_expand(video: str, per_frame: dict[int, str],
                   model_path: str, device: str, output: str) -> None:
    """Run the pose model over the clip and write per-frame GT annotations."""
    from src.pose.estimator import PoseEstimator  # heavy import deferred until needed

    est = PoseEstimator(model_path, conf=0.5, device=device)
    cap = open_capture(video)
    anns: list[dict] = []
    fid = 0

    print("\nPhase B — running pose model to fill bbox/keypoints...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        poses = est.estimate(frame, track=True)
        if poses:
            # Dominant person = largest bbox area (clips are single-operator)
            pose = max(poses, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))
            anns.append({
                "frame_id": fid,
                "person_id": int(pose.person_id),
                "keypoints": pose.keypoints.astype(float).round(1).tolist(),
                "bbox": [round(float(v), 1) for v in pose.bbox],
                "posture": per_frame.get(fid, "normal"),
            })
        fid += 1
        if fid % 30 == 0:
            print(f"  ...{fid} frames processed", end="\r", flush=True)
    cap.release()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(anns, f, indent=2)

    counts = Counter(a["posture"] for a in anns)
    print(f"\nWrote {len(anns)} annotations ({fid} frames, "
          f"{fid - len(anns)} with no person detected) -> {output}")
    for k, v in counts.items():
        print(f"  {k:<28} {v} frames")


def main() -> None:
    args = parse_args()
    per_frame = phase_a_label(args.video)
    if not per_frame:
        print("No frames labelled — aborting.")
        return
    phase_b_expand(args.video, per_frame, args.pose_model, args.device, args.output)


if __name__ == "__main__":
    main()
