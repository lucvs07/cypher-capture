"""
Unified inference pipeline — single capture loop feeding both EPI detection and pose estimation.

Latency management: if combined inference exceeds max_latency_ms, EPI model skips frames
(runs every N frames) while pose always runs, preserving real-time skeleton tracking.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from typing import Optional

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from ..pose.estimator import PersonPose, PoseEstimator
from ..pose.angles import extract_angles
from ..pose.reba import compute_reba
from ..pose.classifier import ClassifierThresholds, PostureCategory, PostureClassifier
from ..pose.temporal import TemporalAnalyzer
from ..zones.manager import ZoneManager
from ..zones.detector import ZoneDetector, ZoneStatus
from ..fusion.engine import DecisionEngine, PersonEvent
from ..utils.pose_visualization import draw_pose_overlay
from ..utils.video import open_capture


@dataclass
class PipelineConfig:
    epi_model_path: str = "models/exp.pt"
    pose_model_path: str = "yolo11s-pose.pt"
    device: str = "cpu"
    conf_epi: float = 0.5
    conf_pose: float = 0.5
    zones_config: str = "config/zones/zones.json"
    target_fps: float = 15.0
    max_latency_ms: float = 200.0
    epi_stride_max: int = 4
    approach_buffer_px: float = 80.0
    enter_frames: int = 5
    enter_frames_risk: int = 2
    exit_frames: int = 15
    thresholds: ClassifierThresholds = field(default_factory=ClassifierThresholds)

    @classmethod
    def from_yaml(cls, path: str = "config/pose_config.yaml") -> "PipelineConfig":
        """Build a config from the YAML file; unknown keys are ignored."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        flat: dict = {}
        for section in ("pipeline", "classifier", "zones"):
            flat.update(raw.get(section) or {})

        known = {f.name for f in fields(cls)} - {"thresholds"}
        config = cls(**{k: v for k, v in flat.items() if k in known})

        th_known = {f.name for f in fields(ClassifierThresholds)}
        th_raw = raw.get("thresholds") or {}
        config.thresholds = ClassifierThresholds(
            **{k: v for k, v in th_raw.items() if k in th_known}
        )
        return config


class UnifiedPipeline:
    """
    Frame-level loop: capture → EPI + pose → angles → REBA → classify → zone check → fuse → event.

    EPI and pose run in the same thread sequentially.  If latency exceeds budget,
    _epi_stride is incremented so EPI runs less frequently until within budget.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._pose = PoseEstimator(config.pose_model_path, config.conf_pose, config.device)
        self._classifier = PostureClassifier(
            fps=config.target_fps,
            thresholds=config.thresholds,
            enter_frames=config.enter_frames,
            exit_frames=config.exit_frames,
            enter_frames_risk=config.enter_frames_risk,
        )
        self._temporal = TemporalAnalyzer(fps=config.target_fps)
        self._zones = ZoneManager(config.zones_config)
        self._zone_detector = ZoneDetector(self._zones, config.approach_buffer_px)
        self._decision = DecisionEngine()
        self._epi_model: Optional[YOLO] = None
        self._frame_idx = 0
        self._epi_stride = 1           # EPI runs every N frames
        self._last_epi_bboxes: list[tuple] = []
        self._last_epi_labels: list[str] = []
        self.last_poses: list[PersonPose] = []  # poses from the most recent frame
        self.last_timings: dict[str, float] = {}  # per-stage latency of the last frame (ms)

    def _ensure_epi_model(self) -> None:
        if self._epi_model is None:
            self._epi_model = YOLO(self.config.epi_model_path)

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[PersonEvent]]:
        t0 = time.perf_counter()
        self._frame_idx += 1
        self._ensure_epi_model()

        # Pose runs every frame (keypoints needed for temporal analysis)
        t_pose = time.perf_counter()
        poses = self._pose.estimate(frame, track=True)
        self.last_poses = poses
        pose_ms = (time.perf_counter() - t_pose) * 1000

        # EPI may run at reduced frequency if latency budget is exceeded
        epi_ran = self._frame_idx % self._epi_stride == 0
        epi_ms = 0.0
        if epi_ran:
            t_epi = time.perf_counter()
            assert self._epi_model is not None
            epi_results = self._epi_model(
                frame, conf=self.config.conf_epi, device=self.config.device, verbose=False
            )
            bboxes: list[tuple] = []
            labels: list[str] = []
            for r in epi_results:
                for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
                    bboxes.append(tuple(box.cpu().numpy().tolist()))
                    labels.append(r.names[int(cls_id)])
            self._last_epi_bboxes = bboxes
            self._last_epi_labels = labels
            epi_ms = (time.perf_counter() - t_epi) * 1000

        t_analysis = time.perf_counter()
        person_bboxes = {p.person_id: p.bbox for p in poses}
        epi_assoc = self._decision.associate_epis(
            person_bboxes, self._last_epi_bboxes, self._last_epi_labels
        )

        events: list[PersonEvent] = []
        for pose in poses:
            angles = extract_angles(pose)
            reba = compute_reba(angles)

            feet = [pt for pt in [pose.kp("left_ankle"), pose.kp("right_ankle")] if pt]
            zone_results = self._zone_detector.check(pose.person_id, feet)
            # Only actual zone *entry* drives the CRITICAL_ZONE posture state;
            # APPROACHING results still reach the decision engine, which maps
            # them to their own (lower) severity.
            in_zone = any(z.status == ZoneStatus.INSIDE for z in zone_results)

            # Temporal metrics use the previous frame's confirmed state (one
            # frame of lag) so the fall-risk score can feed this frame's rules.
            prev_state = self._classifier.get_state(pose.person_id)
            metrics = self._temporal.update(
                pose.person_id, angles, pose.bbox,
                is_bad_posture=prev_state.category != PostureCategory.NORMAL,
            )

            posture_state = self._classifier.update(
                pose.person_id, angles, reba, pose.bbox, in_zone,
                fall_risk=metrics.fall_risk_score,
            )

            epi_labels = epi_assoc.get(pose.person_id, [])
            event = self._decision.fuse(
                pose.person_id, posture_state, reba, zone_results, epi_labels
            )
            events.append(event)
        analysis_ms = (time.perf_counter() - t_analysis) * 1000

        t_overlay = time.perf_counter()
        annotated = draw_pose_overlay(frame, poses, events, self._zones.get_all())
        overlay_ms = (time.perf_counter() - t_overlay) * 1000

        elapsed_ms = (time.perf_counter() - t0) * 1000
        # Per-stage timings for the CLI HUD / latency report (Day-1 instrumentation)
        self.last_timings = {
            "pose_ms": pose_ms,
            "epi_ms": epi_ms,
            "epi_ran": epi_ran,
            "analysis_ms": analysis_ms,
            "overlay_ms": overlay_ms,
            "total_ms": elapsed_ms,
        }

        if elapsed_ms > self.config.max_latency_ms and self._epi_stride < self.config.epi_stride_max:
            self._epi_stride += 1
        elif elapsed_ms < self.config.max_latency_ms * 0.7 and self._epi_stride > 1:
            # recover when latency drops well below budget (the 0.7 margin
            # avoids oscillating around the threshold)
            self._epi_stride -= 1

        return annotated, events

    def run_video(self, source: int | str, display: bool = True) -> None:
        cap = open_capture(source)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, _ = self.process_frame(frame)
                if display:
                    cv2.imshow("Cypher — Pose + EPI", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
