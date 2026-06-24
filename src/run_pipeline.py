#!/usr/bin/env python3
"""
CLI entry-point for the unified EPI + pose pipeline (Phase 2).

Drives UnifiedPipeline frame by frame so it can: save annotated video, run
headless, print a per-stage latency HUD, and surface events as they happen
(UnifiedPipeline.run_video discards events and has no output/HUD — this loop
supersedes it for live testing and measurement).

Usage:
    # webcam, headless (for measurement scripts on a GUI-less box)
    python src/run_pipeline.py --source 0 --no-display

    # video file, save annotated output, show window
    python src/run_pipeline.py --source data/validation/cenario1.mp4 \
        --output outputs/captures/cenario1_annot.mp4

    # RTSP stream with a custom zones file
    python src/run_pipeline.py --source rtsp://... --zones config/zones/zones.json
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

# Ensure project root is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.fusion.pipeline import PipelineConfig, UnifiedPipeline
from src.utils.pose_visualization import draw_epi_boxes
from src.utils.video import open_capture


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cypher — unified EPI + pose pipeline")
    p.add_argument("--source", required=True,
                   help="webcam index (e.g. 0) | video path | RTSP URL")
    p.add_argument("--config", default="config/pose_config.yaml",
                   help="pose_config.yaml path (default: config/pose_config.yaml)")
    p.add_argument("--zones", default=None,
                   help="override zones JSON path (default: taken from config)")
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"],
                   help="override inference device (default: taken from config; "
                        "use 'cuda' to meet the 200ms latency target)")
    p.add_argument("--output", default=None,
                   help="path to save annotated video (e.g. outputs/captures/run.mp4)")
    p.add_argument("--no-display", action="store_true",
                   help="run headless — no GUI window (for measurement / servers)")
    p.add_argument("--min-severity", type=int, default=2,
                   help="print events at/above this severity (0-4, default 2)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after N frames (0 = until source ends / 'q')")
    p.add_argument("--show-epi", action="store_true",
                   help="dev: draw the raw EPI detection boxes from the PPE model")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = PipelineConfig.from_yaml(args.config)
    if args.zones:
        config.zones_config = args.zones
    if args.device:
        config.device = args.device

    print(f"[config] device={config.device}  pose={config.pose_model_path}  epi={config.epi_model_path}")
    print(f"[config] zones={config.zones_config}  target_fps={config.target_fps}  budget={config.max_latency_ms:.0f}ms")

    pipeline = UnifiedPipeline(config)
    print(f"[zones]  {len(pipeline._zones.get_all())} zone(s) loaded")

    cap = open_capture(args.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    src_fps = cap.get(cv2.CAP_PROP_FPS) or config.target_fps

    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps, (width, height))
        print(f"[output] writing annotated video to {args.output} ({width}x{height} @ {src_fps:.0f}fps)")

    window = "Cypher — EPI + Pose (q to quit)"
    if not args.no_display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    roll: deque[float] = deque(maxlen=30)   # rolling window for live FPS
    all_lat: list[float] = []               # full-run latencies for final stats
    frame_idx = 0
    t_start = time.perf_counter()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            annotated, events = pipeline.process_frame(frame)
            dt_ms = (time.perf_counter() - t0) * 1000

            if args.show_epi:
                annotated = draw_epi_boxes(
                    annotated, pipeline._last_epi_bboxes, pipeline._last_epi_labels
                )

            roll.append(dt_ms)
            all_lat.append(dt_ms)
            frame_idx += 1

            tm = pipeline.last_timings
            avg = sum(roll) / len(roll)
            fps = 1000.0 / avg if avg > 0 else 0.0
            epi_tag = "" if tm.get("epi_ran") else "(skip)"
            hud = (f"f{frame_idx:5d}  {dt_ms:6.1f}ms (avg {avg:5.1f} | {fps:4.1f} FPS)  "
                   f"pose {tm.get('pose_ms', 0):5.1f} | epi {tm.get('epi_ms', 0):5.1f}{epi_tag} | "
                   f"analysis {tm.get('analysis_ms', 0):4.1f} | overlay {tm.get('overlay_ms', 0):4.1f} | "
                   f"stride {pipeline._epi_stride}")
            print(hud, end="\r", flush=True)

            # Event sink: surface anything at/above the severity threshold
            for ev in events:
                if ev.severity >= args.min_severity:
                    present = ",".join(ev.epi_present) or "-"
                    viol = ",".join(ev.epi_violations) or "-"
                    print(f"\n[EVENT] f{frame_idx} id#{ev.person_id} sev={ev.severity} "
                          f"posture={ev.posture} reba={ev.reba_score}(AL{ev.action_level}) "
                          f"zone={ev.zone_status}/{ev.zone_name or '-'} ppe=[{present}] viol={viol} :: {ev.reason}")

            # Burn a compact HUD onto the saved/displayed frame too
            cv2.putText(annotated, f"{dt_ms:.0f}ms  {fps:.1f}FPS  stride={pipeline._epi_stride}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow(window, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    wall = time.perf_counter() - t_start
    lat = np.array(all_lat) if all_lat else np.array([0.0])
    eff_fps = frame_idx / wall if wall > 0 else 0.0
    p95 = float(np.percentile(lat, 95))
    print(f"\n\n{'=' * 60}")
    print(f"  Frames         : {frame_idx}")
    print(f"  Wall clock     : {wall:.1f}s  ({eff_fps:.1f} FPS effective)")
    print(f"  E2E latency    : mean={lat.mean():.1f}ms  p50={np.percentile(lat, 50):.1f}ms  "
          f"p95={p95:.1f}ms  max={lat.max():.1f}ms")
    target = config.max_latency_ms
    print(f"  Latency target : {'PASS ✓' if p95 <= target else 'FAIL ✗'}  (p95 ≤ {target:.0f}ms)")
    if args.output:
        print(f"  Saved          : {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
