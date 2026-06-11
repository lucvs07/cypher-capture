#!/usr/bin/env python3
"""
Benchmark YOLO-pose model latency and throughput on a video source.

Usage:
    python scripts/benchmark_pose.py --source <video_path|0> [--model yolo11s-pose.pt] [--frames 100]
"""
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pose model latency benchmark")
    p.add_argument("--source", default="0", help="Video file path or webcam index")
    p.add_argument("--model", default="yolo11s-pose.pt", help="YOLO-pose weights")
    p.add_argument("--frames", type=int, default=100, help="Number of frames to measure")
    p.add_argument("--device", default="cpu", help="Inference device: cpu | cuda | mps")
    p.add_argument("--track", action="store_true", help="Enable ByteTrack (adds overhead)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: could not open source '{args.source}'")
        return

    latencies_ms: list[float] = []
    person_counts: list[int] = []
    t_wall_start = time.perf_counter()
    frame_count = 0

    while cap.isOpened() and frame_count < args.frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        if args.track:
            results = model.track(frame, device=args.device, persist=True, verbose=False)
        else:
            results = model(frame, device=args.device, verbose=False)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        n_persons = sum(len(r.boxes) for r in results if r.boxes is not None)
        person_counts.append(n_persons)
        frame_count += 1

    cap.release()

    if not latencies_ms:
        print("No frames processed.")
        return

    wall_s = time.perf_counter() - t_wall_start
    lats = np.array(latencies_ms)

    print(f"\n{'='*48}")
    print(f"  Model      : {args.model}")
    print(f"  Device     : {args.device}")
    print(f"  Tracking   : {args.track}")
    print(f"  Frames     : {frame_count}")
    print(f"  FPS (wall) : {frame_count / wall_s:.1f}")
    print(f"  Latency ms : mean={lats.mean():.1f}  p50={np.percentile(lats,50):.1f}  "
          f"p95={np.percentile(lats,95):.1f}  max={lats.max():.1f}")
    print(f"  Persons    : mean={np.mean(person_counts):.1f} max={max(person_counts)}")
    print(f"  Target ≤200ms: {'PASS ✓' if np.percentile(lats,95) <= 200 else 'FAIL ✗'}")
    print(f"{'='*48}\n")


if __name__ == "__main__":
    main()
