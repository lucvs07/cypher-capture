#!/usr/bin/env python3
"""
Evaluation script for pose estimation and postural classification (Phase 2, Etapa 8).

Metrics computed:
  - PCK@0.1  : Percent Correct Keypoints at threshold = 0.1 × person_height
  - Accuracy  : per posture category and overall
  - Recall    : for immediate_risk (target ≥ 0.90)
  - False alarm rate  : alert episodes per hour, measured ONLY on frames
    annotated as normal (unannotated frames are excluded — absence of a
    label is not evidence of absence of an event)
  - E2E latency       : mean and p95, excluding the model warm-up frame

Ground-truth JSON format (list of frame annotations):
  [
    {
      "frame_id": 0,
      "person_id": 1,
      "keypoints": [[x,y], ...],   // 17 pairs; [0,0] = unlabelled
      "bbox": [x1, y1, x2, y2],
      "posture": "normal"          // normal | ergonomically_inadequate | immediate_risk | critical_zone
    },
    ...
  ]

Usage:
    python scripts/evaluate_pose.py \\
        --annotations data/annotations/gt.json \\
        --video data/validation/video.mp4 \\
        [--pose-model yolo11s-pose.pt] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion.pipeline import PipelineConfig, UnifiedPipeline
from src.utils.video import open_capture


CATEGORIES = ["normal", "ergonomically_inadequate", "immediate_risk", "critical_zone"]

MATCH_IOU = 0.3  # min bbox IoU to match a predicted pose to a GT annotation


def _bbox_iou(a: list, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def compute_pck(
    pred_kps: list[np.ndarray],
    gt_kps: list[np.ndarray],
    bbox_heights: list[float],
    threshold: float = 0.1,
) -> float:
    correct = total = 0
    for pred, gt, h in zip(pred_kps, gt_kps, bbox_heights):
        dist = np.linalg.norm(pred - gt, axis=1)
        thresh_px = threshold * h
        valid = (gt[:, 0] > 0) | (gt[:, 1] > 0)  # labelled keypoints
        correct += int(np.sum((dist < thresh_px) & valid))
        total += int(np.sum(valid))
    return correct / total if total > 0 else 0.0


def compute_classification_metrics(
    predictions: list[str],
    ground_truth: list[str],
) -> dict:
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    for pred, gt in zip(predictions, ground_truth):
        for cat in CATEGORIES:
            pred_pos = pred == cat
            gt_pos = gt == cat
            if pred_pos and gt_pos:
                tp[cat] += 1
            elif pred_pos and not gt_pos:
                fp[cat] += 1
            elif not pred_pos and gt_pos:
                fn[cat] += 1

    metrics: dict = {}
    for cat in CATEGORIES:
        prec = tp[cat] / (tp[cat] + fp[cat]) if (tp[cat] + fp[cat]) > 0 else 0.0
        rec = tp[cat] / (tp[cat] + fn[cat]) if (tp[cat] + fn[cat]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[cat] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}

    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    metrics["accuracy"] = round(correct / len(predictions), 3) if predictions else 0.0
    return metrics


def run_evaluation(args: argparse.Namespace) -> None:
    gt_path = Path(args.annotations)
    if not gt_path.exists():
        print(f"ERROR: annotations file not found: {gt_path}")
        sys.exit(1)

    with open(gt_path) as f:
        gt_data: list[dict] = json.load(f)

    print(f"Loaded {len(gt_data)} ground-truth annotations from {gt_path}")

    config = PipelineConfig(
        pose_model_path=args.pose_model,
        device=args.device,
        conf_pose=0.5,
        zones_config=args.zones,
    )
    pipeline = UnifiedPipeline(config)

    try:
        cap = open_capture(args.video)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 15.0

    gt_index: dict[int, list[dict]] = defaultdict(list)
    for ann in gt_data:
        gt_index[ann["frame_id"]].append(ann)

    latencies_ms: list[float] = []
    pred_kps_list: list[np.ndarray] = []
    gt_kps_list: list[np.ndarray] = []
    bbox_heights: list[float] = []
    pred_labels: list[str] = []
    gt_labels: list[str] = []

    # False alarms: episode-based (a sustained alert counts once, not once
    # per frame), counted only on frames whose GT says everyone is normal.
    false_alarm_episodes = 0
    normal_frame_count = 0
    alerting_persons: set[int] = set()

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        annotated, events = pipeline.process_frame(frame)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        frame_anns = gt_index.get(frame_id, [])
        event_map = {e.person_id: e for e in events}
        poses = pipeline.last_poses

        for ann in frame_anns:
            gt_labels.append(ann["posture"])

            # Match GT annotation to a predicted pose by bbox IoU (GT
            # person_id and tracker IDs are independent numbering schemes)
            best_pose, best_iou = None, MATCH_IOU
            for pose in poses:
                iou = _bbox_iou(ann["bbox"], pose.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_pose = pose

            if best_pose is not None:
                pred_labels.append(
                    event_map[best_pose.person_id].posture
                    if best_pose.person_id in event_map else "normal"
                )
                # PCK sample
                pred_kps_list.append(best_pose.keypoints.copy())
                gt_kps_list.append(np.array(ann["keypoints"], dtype=np.float32))
                bbox_heights.append(ann["bbox"][3] - ann["bbox"][1])
            else:
                # Missed detection counts against classification too
                pred_labels.append("normal" if ann["posture"] == "normal" else "missed")

        # False-alarm accounting on annotated-normal frames only
        if frame_anns and all(a["posture"] == "normal" for a in frame_anns):
            normal_frame_count += 1
            for ev in events:
                if ev.severity >= 2:
                    if ev.person_id not in alerting_persons:
                        false_alarm_episodes += 1
                        alerting_persons.add(ev.person_id)
                else:
                    alerting_persons.discard(ev.person_id)

        frame_id += 1

    cap.release()

    # First frame includes one-time model loading; report it separately
    warmup_ms = latencies_ms[0] if latencies_ms else 0.0
    steady = np.array(latencies_ms[1:]) if len(latencies_ms) > 1 else np.array([0.0])
    cls_metrics = compute_classification_metrics(pred_labels, gt_labels)
    pck = compute_pck(pred_kps_list, gt_kps_list, bbox_heights)

    normal_hours = normal_frame_count / fps_video / 3600.0
    false_alarm_rate = false_alarm_episodes / normal_hours if normal_hours > 0 else float("nan")

    print(f"\n{'='*56}")
    print(f"  Frames processed : {frame_id}")
    print(f"  Warm-up frame    : {warmup_ms:.0f} ms (excluded from stats)")
    print(f"  Latency ms       : mean={steady.mean():.1f}  p95={np.percentile(steady,95):.1f}")
    print(f"  Latency target   : {'PASS ✓' if np.percentile(steady,95)<=200 else 'FAIL ✗'} (≤200ms)")
    print()
    print(f"  PCK@0.1          : {pck:.3f} on {len(pred_kps_list)} matched persons "
          f"({'PASS ✓' if pck >= 0.80 else 'FAIL ✗'}, target ≥0.80)")
    print(f"  Overall accuracy : {cls_metrics['accuracy']:.3f}  (target ≥0.85)")
    print()
    for cat in CATEGORIES:
        m = cls_metrics[cat]
        flag = ""
        if cat == "immediate_risk":
            flag = "  ← target recall ≥0.90 " + ("✓" if m["recall"] >= 0.90 else "✗")
        print(f"  {cat:<30} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}{flag}")
    print()
    if normal_hours > 0:
        print(f"  False alarms     : {false_alarm_episodes} episode(s) in "
              f"{normal_frame_count} annotated-normal frames "
              f"(~{false_alarm_rate:.1f}/h, target → 0)")
    else:
        print("  False alarms     : n/a — no frames annotated as normal")
    print(f"{'='*56}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--pose-model", default="yolo11s-pose.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--zones", default="config/zones/zones.json",
                   help="zones JSON (needed to score critical_zone scenarios)")
    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
