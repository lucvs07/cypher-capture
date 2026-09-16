from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ultralytics import YOLO

COCO_KEYPOINTS = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

KP: dict[str, int] = {name: idx for idx, name in enumerate(COCO_KEYPOINTS)}


@dataclass
class PersonPose:
    person_id: int
    keypoints: np.ndarray           # (17, 2) pixel coordinates
    confidences: np.ndarray         # (17,) float in [0, 1]
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    timestamp: float = field(default_factory=time.time)
    kp_confidence_threshold: float = 0.5

    def kp(self, name: str) -> Optional[tuple[float, float]]:
        """Return (x, y) for the keypoint if confidence >= threshold, else None."""
        idx = KP[name]
        if self.confidences[idx] >= self.kp_confidence_threshold:
            return (float(self.keypoints[idx, 0]), float(self.keypoints[idx, 1]))
        return None

    @property
    def bbox_height(self) -> float:
        return max(self.bbox[3] - self.bbox[1], 1.0)

    @property
    def bbox_width(self) -> float:
        return max(self.bbox[2] - self.bbox[0], 1.0)


class PoseEstimator:
    """Thin wrapper around YOLO11-pose with ByteTrack integration."""

    def __init__(
        self,
        model_path: str = "yolo11s-pose.pt",
        conf: float = 0.5,
        device: str = "cpu",
        kp_confidence: float = 0.5,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        self.kp_confidence = kp_confidence

    def estimate(self, frame: np.ndarray, track: bool = True) -> list[PersonPose]:
        """Run pose estimation on a single frame. Returns one PersonPose per person."""
        if track:
            results = self.model.track(
                frame,
                conf=self.conf,
                device=self.device,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
            )
        else:
            results = self.model(frame, conf=self.conf, device=self.device, verbose=False)

        poses: list[PersonPose] = []
        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue

            kps_data = r.keypoints.data.cpu().numpy()  # (N, 17, 3)
            boxes = r.boxes.xyxy.cpu().numpy()          # (N, 4)
            ids = r.boxes.id

            for i in range(len(kps_data)):
                # ByteTrack IDs are positive; use negative fallback IDs for
                # not-yet-confirmed tracks so they can never collide with (and
                # poison the per-person state of) a real track ID.
                person_id = int(ids[i]) if ids is not None else -(i + 1)
                poses.append(PersonPose(
                    person_id=person_id,
                    keypoints=kps_data[i, :, :2],
                    confidences=kps_data[i, :, 2],
                    bbox=tuple(boxes[i].tolist()),
                    kp_confidence_threshold=self.kp_confidence,
                ))
        return poses
