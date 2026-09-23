from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..pose.classifier import PostureCategory, PostureState
from ..pose.epi_catalog import EpiTipo
from ..pose.reba import REBAScore
from ..zones.detector import ZoneResult, ZoneStatus


@dataclass
class PersonEvent:
    """Unified per-person event emitted each frame. Schema targets Phase 3 persistence."""
    timestamp: float = field(default_factory=time.time)
    person_id: int = 0
    epi_violations: list[EpiTipo] = field(default_factory=list)  # EPI exigido confirmado ausente
    epi_present: list[EpiTipo] = field(default_factory=list)     # EPI detectado nesta pessoa
    posture: str = "normal"
    previous_posture: str = "normal"   # categoria confirmada no frame anterior
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

    @staticmethod
    def worst_zone(zone_results: list[ZoneResult]) -> Optional[ZoneResult]:
        """Zona mais severa para esta pessoa neste frame (inside > approaching,
        maior severidade desempata). Extraido de fuse() pra ser reusado pelo
        pipeline na resolucao de escopo de EPI (zona ∪ checklist global,
        achado 2026-09-20) sem duplicar o criterio de desempate."""
        if not zone_results:
            return None
        return max(zone_results, key=lambda z: (z.status == ZoneStatus.INSIDE, z.severity))

    def fuse(
        self,
        person_id: int,
        posture_state: PostureState,
        reba: REBAScore,
        zone_results: list[ZoneResult],
        epi_present: list[EpiTipo],
        epi_missing_confirmed: list[EpiTipo] | None = None,
    ) -> PersonEvent:
        event = PersonEvent(person_id=person_id)
        event.posture = posture_state.category.value
        event.reason = posture_state.reason
        event.reba_score = reba.final_score
        event.action_level = reba.action_level

        # O modelo so tem classes de presenca (ver epi_catalog.py): violacao
        # e sempre a inferencia de ausencia por debounce (epi_absence.py,
        # achado 2026-09-20). Deduplicado, ordem preservada.
        event.epi_violations = list(dict.fromkeys(epi_missing_confirmed or []))
        event.epi_present = list(dict.fromkeys(epi_present))

        # Zone: report worst (inside > approaching, higher severity wins)
        worst = self.worst_zone(zone_results)
        if worst is not None:
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
        # So zona "restricted" eleva severidade por presenca (achado de campo
        # 2026-09-20 — "attention"/area de trabalho nao e zona de risco, so
        # existe pra restringir o que e avaliado via delimita_avaliacao).
        if event.zone_status == "inside" and worst is not None and worst.zone_type == "restricted":
            sev = max(sev, 3)
        elif event.zone_status == "approaching":
            sev = max(sev, 1)
        if event.epi_violations:
            sev = max(sev, 2)
        if reba.action_level >= 4:
            sev = max(sev, 3)

        # Taxonomy §4.2: being in/near a zone while missing a PPE the zone
        # requires escalates severity by +1 (capped at 4). required_ppe ja
        # chega resolvido pra EpiTipo desde a carga da zona, entao e
        # pertinencia direta — sem montar string (o f"no_{ppe}" antigo nunca
        # batia com o nome canonico do cadastro).
        if worst is not None and any(t in worst.required_ppe for t in event.epi_violations):
            sev = min(sev + 1, 4)

        event.severity = sev
        return event

    @staticmethod
    def associate_epis(
        person_bboxes: dict[int, tuple],
        epi_bboxes: list[tuple],
        epi_labels: list[str],
        containment_threshold: float = 0.5,
    ) -> dict[int, list[EpiTipo]]:
        """
        Assign each EPI detection to the person whose bbox covers most of it.
        An EPI box is associated when at least containment_threshold of its
        area lies inside the person bbox; ties go to the highest coverage.

        ``epi_labels`` sao os nomes crus do detector — unico ponto do fluxo
        que os ve. Sai ``EpiTipo``; ``Person`` e classe desconhecida sao
        descartadas aqui.
        """
        result: dict[int, list[EpiTipo]] = {pid: [] for pid in person_bboxes}
        for epi_box, label in zip(epi_bboxes, epi_labels):
            tipo = EpiTipo.from_detector_label(label)
            if tipo is None:
                continue
            best_pid, best_cov = None, containment_threshold
            for pid, p_box in person_bboxes.items():
                cov = _containment(p_box, epi_box)
                if cov > best_cov:
                    best_cov = cov
                    best_pid = pid
            if best_pid is not None:
                result[best_pid].append(tipo)
        return result
