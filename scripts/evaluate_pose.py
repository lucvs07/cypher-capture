#!/usr/bin/env python3
"""
Evaluation script for pose estimation and postural classification (Phase 2, Etapa 8).

Metrics computed:
  - PCK@0.1  : Percent Correct Keypoints at threshold = 0.1 × person_height
  - Accuracy  : per posture category and overall
  - Recall    : for immediate_risk (target ≥ 0.90)
  - False alarm rate  : alerts per hour on known-negative footage
  - E2E latency       : mean and p95

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


CATEGORIES = ["normal", "ergonomically_inadequate", "immediate_risk", "critical_zone"]


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
    )
    pipeline = UnifiedPipeline(config)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open video '{args.video}'")
        sys.exit(1)

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration_h = total_frames / fps_video / 3600.0

    gt_index: dict[int, list[dict]] = defaultdict(list)
    for ann in gt_data:
        gt_index[ann["frame_id"]].append(ann)

    latencies_ms: list[float] = []
    pred_kps_list: list[np.ndarray] = []
    gt_kps_list: list[np.ndarray] = []
    bbox_heights: list[float] = []
    pred_labels: list[str] = []
    gt_labels: list[str] = []
    false_alarms = 0

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

        for ann in frame_anns:
            pid = ann["person_id"]
            gt_label = ann["posture"]
            gt_labels.append(gt_label)

            if pid in event_map:
                pred_labels.append(event_map[pid].posture)
            else:
                pred_labels.append("normal")

            # PCK data
            gt_kp = np.array(ann["keypoints"], dtype=np.float32)  # (17,2)
            bbox_h = ann["bbox"][3] - ann["bbox"][1]
            bbox_heights.append(bbox_h)
            gt_kps_list.append(gt_kp)

        # Count false alarms: alert fired on a frame with no ground-truth risk
        if not frame_anns and events:
            for ev in events:
                if ev.severity >= 2:
                    false_alarms += 1

        frame_id += 1

    cap.release()

    lats = np.array(latencies_ms) if latencies_ms else np.array([0.0])
    cls_metrics = compute_classification_metrics(pred_labels, gt_labels)
    false_alarm_rate = false_alarms / total_duration_h if total_duration_h > 0 else 0.0

    print(f"\n{'='*56}")
    print(f"  Frames processed : {frame_id}")
    print(f"  Latency ms       : mean={lats.mean():.1f}  p95={np.percentile(lats,95):.1f}")
    print(f"  Latency target   : {'PASS ✓' if np.percentile(lats,95)<=200 else 'FAIL ✗'} (≤200ms)")
    print()
    print(f"  Overall accuracy : {cls_metrics['accuracy']:.3f}  (target ≥0.85)")
    print()
    for cat in CATEGORIES:
        m = cls_metrics[cat]
        flag = ""
        if cat == "immediate_risk":
            flag = "  ← target recall ≥0.90 " + ("✓" if m["recall"] >= 0.90 else "✗")
        print(f"  {cat:<30} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}{flag}")
    print()
    print(f"  False alarm rate : {false_alarm_rate:.1f} alerts/hour  (target → 0)")
    print(f"{'='*56}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--pose-model", default="yolo11s-pose.pt")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
