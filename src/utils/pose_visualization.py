from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ..fusion.engine import PersonEvent
    from ..zones.manager import Zone

from ..pose.estimator import COCO_KEYPOINTS, KP, PersonPose
from .visualization import class_color

# Skeleton connections as (keypoint_name_a, keypoint_name_b) pairs
SKELETON = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

# BGR severity colours
SEVERITY_COLOR: dict[int, tuple[int, int, int]] = {
    0: (0, 200, 0),     # green — ok
    1: (0, 200, 200),   # yellow — low
    2: (0, 130, 255),   # orange — medium
    3: (0, 0, 220),     # red — high
    4: (30, 0, 160),    # dark red — critical
}

ZONE_COLORS = {
    "restricted": (0, 0, 200),
    "attention": (0, 140, 255),
}


def draw_pose_overlay(
    frame: np.ndarray,
    poses: list[PersonPose],
    events: list["PersonEvent"],
    zones: list["Zone"],
) -> np.ndarray:
    out = frame.copy()

    if zones:
        zone_draw = [
            (np.array(z.polygon, dtype=np.int32),
             ZONE_COLORS.get(z.zone_type, (0, 0, 200)),
             z.name)
            for z in zones
        ]
        # Semi-transparent zone fill (full-frame blend only when zones exist)
        overlay = out.copy()
        for pts, color, _ in zone_draw:
            cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)

        # Zone borders and labels
        for pts, color, name in zone_draw:
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(out, name, pts[0].tolist(), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    event_map: dict[int, "PersonEvent"] = {e.person_id: e for e in events}

    for pose in poses:
        ev = event_map.get(pose.person_id)
        sev = ev.severity if ev else 0
        color = SEVERITY_COLOR.get(sev, (0, 200, 0))

        # Skeleton lines
        for a_name, b_name in SKELETON:
            a_idx, b_idx = KP[a_name], KP[b_name]
            if pose.confidences[a_idx] >= 0.5 and pose.confidences[b_idx] >= 0.5:
                pa = (int(pose.keypoints[a_idx, 0]), int(pose.keypoints[a_idx, 1]))
                pb = (int(pose.keypoints[b_idx, 0]), int(pose.keypoints[b_idx, 1]))
                cv2.line(out, pa, pb, color, 2)

        # Keypoint dots
        for i in range(len(COCO_KEYPOINTS)):
            if pose.confidences[i] >= 0.5:
                pt = (int(pose.keypoints[i, 0]), int(pose.keypoints[i, 1]))
                cv2.circle(out, pt, 4, color, -1)

        # Info label above bbox
        x1, y1 = int(pose.bbox[0]), int(pose.bbox[1])
        if ev:
            parts = [
                f"#{ev.person_id}",
                f"REBA:{ev.reba_score}",
                ev.posture[:8],
            ]
            if ev.zone_status != "outside" and ev.zone_name:
                parts.append(ev.zone_name[:10])
            if ev.epi_present:
                parts.append("ppe:" + ",".join(ev.epi_present))
            label = "  ".join(parts)
        else:
            label = f"#{pose.person_id}"

        label_x = max(x1, 2)
        label_y = max(y1 - 8, 14)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (label_x - 2, label_y - th - 2), (label_x + tw + 2, label_y + 2), (0, 0, 0), -1)
        cv2.putText(out, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return out


def draw_epi_boxes(
    frame: np.ndarray,
    bboxes: list[tuple],
    labels: list[str],
) -> np.ndarray:
    """Dev overlay: draw the raw EPI detection boxes from the PPE model on top
    of the pose overlay (gated behind run_pipeline's --show-epi flag).

    Drawn in place with thin rectangles so they read as a debug layer distinct
    from the thicker, severity-coloured skeletons. Colours match Phase 1
    (class_color): green = PPE present, red = violation, amber = Person.
    """
    for box, label in zip(bboxes, labels):
        x1, y1, x2, y2 = (int(v) for v in box)
        color = class_color(label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, label, (x1 + 2, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return frame
