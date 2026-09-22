"""
Unified inference pipeline — single capture loop feeding both EPI detection and pose estimation.

Latency management: if combined inference exceeds max_latency_ms, EPI model skips frames
(runs every N frames) while pose always runs, preserving real-time skeleton tracking.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field, fields
from typing import Optional

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from ..pose.estimator import PersonPose, PoseEstimator
from ..pose.angles import extract_angles, vertical_from_roll
from ..pose.reba import compute_reba
from ..pose.reba_smoother import RebaSmoother
from ..pose.classifier import ClassifierThresholds, PostureCategory, PostureClassifier
from ..pose.epi_absence import EpiAbsenceTracker
from ..pose.epi_catalog import EpiTipo
from ..pose.temporal import TemporalAnalyzer
from ..zones.manager import ZoneManager
from ..zones.detector import ZoneDetector, ZoneStatus
from ..fusion.engine import DecisionEngine, PersonEvent
from ..fusion.track_state import EsquecedorDeTracks
from ..utils.pose_visualization import draw_pose_overlay
from ..utils.video import open_capture


@dataclass
class PipelineConfig:
    epi_model_path: str = "models/exp.pt"
    pose_model_path: str = "yolo11s-pose.pt"
    device: str = "cpu"
    conf_epi: float = 0.5
    conf_pose: float = 0.5
    kp_confidence: float = 0.5
    # Achado de campo 2026-09-20: o default do Ultralytics (track_buffer=30,
    # ~1.2s a 25 FPS) descarta o track e troca de ID a cada oclusao/blur
    # breve, quebrando tanto a confirmacao de ausencia de EPI (exige frames
    # consecutivos sob o mesmo ID) quanto o REBA suavizado.
    pose_tracker_config: str = "config/bytetrack_tuned.yaml"
    camera_roll_deg: float = 0.0
    fall_require_corroboration: bool = False
    zones_config: str = "config/zones/zones.json"
    target_fps: float = 15.0
    max_latency_ms: float = 200.0
    epi_stride_max: int = 4
    approach_buffer_px: float = 80.0
    enter_frames: int = 5
    enter_frames_risk: int = 2
    exit_frames: int = 15
    fall_confirm_seconds: float = 1.0
    # Ausencia de EPI (achado de campo 2026-09-20): o detector so tem classes
    # de presenca, "ausente" e sempre uma inferencia sobre N frames seguidos
    # sem deteccao (ver pose/epi_absence.py) — 1 frame perdido por oclusao ou
    # blur nao pode virar alerta. epi_checklist_global sao nomes canonicos
    # (EpiTipo, ex. "HELMET") usados quando a pessoa nao esta numa zona com
    # required_ppe proprio (zona tem prioridade sobre o checklist da empresa).
    epi_confirm_seconds: float = 2.0
    epi_min_track_frames: int = 15
    epi_checklist_global: list[str] = field(default_factory=list)
    # Frames sem ver um track ate liberar o estado por pessoa dele (ver
    # fusion/track_state.py). Precisa ser maior que o track_buffer do
    # ByteTrack (90 em bytetrack_tuned.yaml), senao esquece quem so ficou
    # ocluido e ainda vai voltar com o mesmo id.
    forget_track_after_frames: int = 150
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
        self._pose = PoseEstimator(
            config.pose_model_path, config.conf_pose, config.device,
            kp_confidence=config.kp_confidence,
            tracker=config.pose_tracker_config,
        )
        self._classifier = PostureClassifier(
            fps=config.target_fps,
            thresholds=config.thresholds,
            enter_frames=config.enter_frames,
            exit_frames=config.exit_frames,
            enter_frames_risk=config.enter_frames_risk,
            fall_confirm_frames=round(config.fall_confirm_seconds * config.target_fps),
            fall_require_corroboration=config.fall_require_corroboration,
        )
        self._temporal = TemporalAnalyzer(fps=config.target_fps)
        self._vertical_reference = vertical_from_roll(config.camera_roll_deg)
        self._reba_smoother = RebaSmoother(window=5)
        self._indeterminate_history: deque = deque(maxlen=150)  # ~10s a 15fps, so pro HUD
        self._zones = ZoneManager(config.zones_config)
        self._zone_detector = ZoneDetector(self._zones, config.approach_buffer_px)
        self._decision = DecisionEngine()
        self._epi_absence = EpiAbsenceTracker(
            confirm_frames=round(config.epi_confirm_seconds * config.target_fps),
            min_track_frames=config.epi_min_track_frames,
        )
        self._epi_checklist_global = [
            tipo for tipo in (
                EpiTipo.from_wire_name(nome) for nome in config.epi_checklist_global
            ) if tipo is not None
        ]
        self._esquecedor = EsquecedorDeTracks(
            config.forget_track_after_frames,
            [self._classifier.forget, self._temporal.forget,
             self._reba_smoother.forget, self._epi_absence.forget],
        )
        self._epi_model: Optional[YOLO] = None
        self._frame_idx = 0
        self._epi_stride = 1           # EPI runs every N frames
        self._last_epi_bboxes: list[tuple] = []
        self._last_epi_labels: list[str] = []
        self.last_poses: list[PersonPose] = []  # poses from the most recent frame
        self.last_timings: dict[str, float] = {}  # per-stage latency of the last frame (ms)
        self._frame_publisher = None    # callable opcional, alimentado pelo MJPEG

    def _ensure_epi_model(self) -> None:
        if self._epi_model is None:
            self._epi_model = YOLO(self.config.epi_model_path)

    def set_frame_publisher(self, publisher) -> None:
        """Registra quem recebe cada frame anotado (ex.: MjpegServer.publish)."""
        self._frame_publisher = publisher

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

        frame_height, frame_width = frame.shape[:2]
        edge_margin = 5.0  # px — bbox colado na borda = pessoa cortada, nao deitada

        events: list[PersonEvent] = []
        for pose in poses:
            angles = extract_angles(pose, self._vertical_reference)
            reba = compute_reba(angles)
            self._indeterminate_history.append(reba.indeterminate)
            reba.final_score = self._reba_smoother.smooth(pose.person_id, reba)

            feet = [pt for pt in [pose.kp("left_ankle"), pose.kp("right_ankle")] if pt]
            zone_results = self._zone_detector.check(pose.person_id, feet)
            # Only actual zone *entry* drives the CRITICAL_ZONE posture state;
            # APPROACHING results still reach the decision engine, which maps
            # them to their own (lower) severity. So "restricted" abre
            # CRITICAL_ZONE/ZONE_VIOLATION (achado de campo 2026-09-20 —
            # antes uma zona "attention"/area de trabalho disparava o mesmo
            # alerta so por alguem entrar nela, sem checar o tipo).
            in_zone = any(
                z.status == ZoneStatus.INSIDE and z.zone_type == "restricted"
                for z in zone_results
            )
            # Gate de avaliacao (achado SPI 2026-09-16): zonas com
            # delimita_avaliacao=True sao a area de trabalho declarada —
            # existindo ao menos uma na camera, so quem esta dentro dela tem
            # EPI/pose avaliados. Zona de perigo simples (delimita_avaliacao
            # falso, o default) nunca restringe nada, so eleva severidade
            # quando alguem entra. Sem nenhuma zona delimitadora, avalia todo
            # mundo — comportamento anterior preservado.
            avaliacao_habilitada = ZoneDetector.avaliacao_habilitada(
                zone_results, self._zones.get_all())

            # Temporal metrics use the previous frame's confirmed state (one
            # frame of lag) so the fall-risk score can feed this frame's rules.
            prev_state = self._classifier.get_state(pose.person_id)
            prev_category = prev_state.category          # antes do update
            metrics = self._temporal.update(
                pose.person_id, angles, pose.bbox,
                is_bad_posture=prev_state.category != PostureCategory.NORMAL,
            )

            x1, y1, x2, y2 = pose.bbox
            bbox_truncated = (
                x1 <= edge_margin or y1 <= edge_margin
                or x2 >= frame_width - edge_margin or y2 >= frame_height - edge_margin
            )

            posture_state = self._classifier.update(
                pose.person_id, angles, reba, pose.bbox, in_zone,
                fall_risk=metrics.fall_risk_score,
                ergonomic_eval_enabled=avaliacao_habilitada,
                bbox_truncated=bbox_truncated,
            )

            # Mesmo gate desliga EPI: fora da area de trabalho declarada,
            # nao ha avaliacao de EPI a se fazer, so pose ergonomica (achado
            # SPI 2026-09-16 — antes disso EPI rodava pra todo mundo sempre).
            epi_labels = epi_assoc.get(pose.person_id, []) if avaliacao_habilitada else []

            # Escopo do EPI exigido (achado 2026-09-20): a zona onde a pessoa
            # esta, quando exige EPI proprio, tem prioridade sobre o
            # checklist global da empresa — o checklist so vale quando a
            # zona nao resolve nenhum item (sem zona, ou zona sem
            # required_ppe reconhecido).
            epi_missing_confirmed: list[str] = []
            if avaliacao_habilitada:
                worst_zone = DecisionEngine.worst_zone(zone_results)
                required = []
                if worst_zone is not None and worst_zone.required_ppe:
                    required = [
                        t for t in (EpiTipo.from_wire_name(n) for n in worst_zone.required_ppe)
                        if t is not None
                    ]
                if not required:
                    required = self._epi_checklist_global
                if required:
                    epi_missing_confirmed = self._epi_absence.update(
                        pose.person_id, epi_labels, required, bbox_truncated,
                    )

            event = self._decision.fuse(
                pose.person_id, posture_state, reba, zone_results, epi_labels,
                epi_missing_confirmed=epi_missing_confirmed,
            )
            event.previous_posture = prev_category.value
            events.append(event)
        self._esquecedor.update(self._frame_idx, (p.person_id for p in poses))
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

        if self._frame_publisher is not None:
            self._frame_publisher(annotated)

        return annotated, events

    @property
    def indeterminate_rate(self) -> float:
        """Fracao de frames indeterminados na janela recente (HUD de campo:
        subir kp_confidence aumenta isto, e sem esse numero parece que o
        sistema quebrou em vez de estar sendo mais conservador)."""
        if not self._indeterminate_history:
            return 0.0
        return sum(self._indeterminate_history) / len(self._indeterminate_history)

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
