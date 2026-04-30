"""Overlay detection results on frames with PPE-aware color coding."""

import cv2
import numpy as np
from ultralytics.engine.results import Results

# Classes that indicate a violation (missing PPE)
VIOLATION_CLASSES = {"no_helmet", "no_goggle", "no_gloves", "no_boots", "none"}

COLOR_COMPLIANT = (0, 200, 0)    # green
COLOR_VIOLATION = (0, 0, 220)    # red
COLOR_NEUTRAL = (200, 160, 0)    # amber — Person, uncategorized


def class_color(label: str) -> tuple:
    if label in VIOLATION_CLASSES:
        return COLOR_VIOLATION
    if label == "Person":
        return COLOR_NEUTRAL
    return COLOR_COMPLIANT


def draw_detections(frame: np.ndarray, result: Results, conf_threshold: float = 0.5) -> np.ndarray:
    annotated = frame.copy()
    names = result.names

    for box in result.boxes:
        conf = float(box.conf)
        if conf < conf_threshold:
            continue

        cls_id = int(box.cls)
        label = names[cls_id]
        color = class_color(label)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated
