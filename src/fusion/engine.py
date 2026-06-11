from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..pose.classifier import PostureCategory, PostureState
from ..pose.reba import REBAScore
from ..zones.detector import ZoneResult, ZoneStatus


@dataclass
class PersonEvent:
    """Unified per-person event emitted each frame. Schema targets Phase 3 persistence."""
    timestamp: float = field(default_factory=time.time)
    person_id: int = 0
    epi_status: dict[str, bool] = field(default_factory=dict)   # label → compliant
    posture: str = "normal"
    reba_score: int = 1
    action_level: int = 1
    zone_status: str = "outside"
    zone_name: Optional[str] = None
    severity: int = 0    # 0=ok … 4=critical
    reason: str = ""


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class DecisionEngine:
    """Fuses posture, REBA, zone, and EPI information into a per-person event."""

    EPI_VIOLATION_LABELS = {"no_gloves", "no_goggle", "no_helmet", "no_boots", "none"}

    def fuse(
        self,
        person_id: int,
        posture_state: PostureState,
        reba: REBAScore,
        zone_results: list[ZoneResult],
        epi_labels: list[str],
    ) -> PersonEvent:
        event = PersonEvent(person_id=person_id)
        event.posture = posture_state.category.value
        event.reason = posture_state.reason
        event.reba_score = reba.final_score
        event.action_level = reba.action_level

        # EPI compliance map — only track labels that indicate violations
        violations = [lbl for lbl in epi_labels if lbl in self.EPI_VIOLATION_LABELS]
        event.epi_status = {lbl: False for lbl in violations}

        # Zone: report worst (inside > approaching, higher severity wins)
        if zone_results:
            worst = max(
                zone_results,
                key=lambda z: (z.status == ZoneStatus.INSIDE, z.severity),
            )
            event.zone_status = worst.status.value
            event.zone_name = worst.zone_name
        else:
            event.zone_status = "outside"

        # Severity aggregation
        sev = 0
        if posture_state.category == PostureCategory.IMMEDIATE_RISK:
            sev = max(sev, 4)
        elif posture_state.category == PostureCategory.CRITICAL_ZONE:
            sev = max(sev, 3)
        elif posture_state.category == PostureCategory.ERGONOMICALLY_INADEQUATE:
            sev = max(sev, 2)
        if event.zone_status == "inside":
            sev = max(sev, 3)
        elif event.zone_status == "approaching":
            sev = max(sev, 1)
        if violations:
            sev = max(sev, 2)
        if reba.action_level >= 4:
            sev = max(sev, 3)

        event.severity = sev
        return event

    @staticmethod
    def associate_epis(
        person_bboxes: dict[int, tuple],
        epi_bboxes: list[tuple],
        epi_labels: list[str],
        iou_threshold: float = 0.1,
    ) -> dict[int, list[str]]:
        """
        Assign each EPI detection to the person whose bounding box overlaps it most.
        Uses containment-sensitive IoU so small items (helmets) associate correctly.
        """
        result: dict[int, list[str]] = {pid: [] for pid in person_bboxes}
        for epi_box, label in zip(epi_bboxes, epi_labels):
            best_pid, best_iou = None, iou_threshold
            for pid, p_box in person_bboxes.items():
                iou = _iou(p_box, epi_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pid = pid
            if best_pid is not None:
                result[best_pid].append(label)
        return result
