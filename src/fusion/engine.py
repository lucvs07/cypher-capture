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
    epi_violations: list[str] = field(default_factory=list)  # violation labels on this person (no_helmet, …)
    epi_present: list[str] = field(default_factory=list)      # PPE detected on this person (HELMET, VEST, …)
    posture: str = "normal"
    reba_score: int = 1
    action_level: int = 1
    zone_status: str = "outside"
    zone_name: Optional[str] = None
    severity: int = 0    # 0=ok … 4=critical
    reason: str = ""


def _containment(person: tuple, epi: tuple) -> float:
    """Fraction of the EPI box covered by the person box (0–1).

    Plain IoU fails here: a 50x50 helmet fully inside a 200x500 person bbox
    has IoU ≈ 0.025, below any usable threshold. Coverage of the smaller box
    is the right measure for item→person assignment.
    """
    px1, py1, px2, py2 = person
    ex1, ey1, ex2, ey2 = epi
    ix1, iy1 = max(px1, ex1), max(py1, ey1)
    ix2, iy2 = min(px2, ex2), min(py2, ey2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    epi_area = (ex2 - ex1) * (ey2 - ey1)
    return inter / epi_area if epi_area > 0 else 0.0


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

        violations = [lbl for lbl in epi_labels if lbl in self.EPI_VIOLATION_LABELS]
        event.epi_violations = violations
        # PPE actually detected on the person (presence). "Person" is the EPI
        # model's own person class, not equipment, so it is excluded. Deduped,
        # order preserved.
        event.epi_present = list(dict.fromkeys(
            lbl for lbl in epi_labels
            if lbl not in self.EPI_VIOLATION_LABELS and lbl != "Person"
        ))

        # Zone: report worst (inside > approaching, higher severity wins)
        worst = None
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
        if event.epi_violations:
            sev = max(sev, 2)
        if reba.action_level >= 4:
            sev = max(sev, 3)

        # Taxonomy §4.2: being in/near a zone while missing a PPE the zone
        # requires escalates severity by +1 (capped at 4). A required item is
        # "missing" when the matching violation label (e.g. helmet → no_helmet)
        # was associated to this person.
        if worst is not None and worst.required_ppe and event.epi_violations:
            missing = [
                ppe for ppe in worst.required_ppe
                if f"no_{ppe}" in event.epi_violations
            ]
            if missing:
                sev = min(sev + 1, 4)

        event.severity = sev
        return event

    @staticmethod
    def associate_epis(
        person_bboxes: dict[int, tuple],
        epi_bboxes: list[tuple],
        epi_labels: list[str],
        containment_threshold: float = 0.5,
    ) -> dict[int, list[str]]:
        """
        Assign each EPI detection to the person whose bbox covers most of it.
        An EPI box is associated when at least containment_threshold of its
        area lies inside the person bbox; ties go to the highest coverage.
        """
        result: dict[int, list[str]] = {pid: [] for pid in person_bboxes}
        for epi_box, label in zip(epi_bboxes, epi_labels):
            best_pid, best_cov = None, containment_threshold
            for pid, p_box in person_bboxes.items():
                cov = _containment(p_box, epi_box)
                if cov > best_cov:
                    best_cov = cov
                    best_pid = pid
            if best_pid is not None:
                result[best_pid].append(label)
        return result
