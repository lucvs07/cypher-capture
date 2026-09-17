from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from .manager import Zone, ZoneManager


class ZoneStatus(Enum):
    OUTSIDE = "outside"
    APPROACHING = "approaching"   # within buffer distance but not inside
    INSIDE = "inside"


@dataclass
class ZoneResult:
    person_id: int
    zone_id: str
    zone_name: str
    status: ZoneStatus
    distance_px: float
    severity: int
    required_ppe: list[str] = field(default_factory=list)
    delimita_avaliacao: bool = False


APPROACH_BUFFER_PX = 80


class ZoneDetector:
    """Tests zone proximity for a person's feet.

    Returns only INSIDE and APPROACHING results; zones the person is clearly
    outside (beyond approach_buffer_px) are omitted from the result list.
    """

    def __init__(self, manager: ZoneManager, approach_buffer_px: float = APPROACH_BUFFER_PX) -> None:
        self.manager = manager
        self.buffer = approach_buffer_px
        # zones are static after load; cache the ndarray per zone_id instead
        # of rebuilding it persons × zones times per frame
        self._poly_cache: dict[str, np.ndarray] = {}

    def _polygon(self, zone: Zone) -> np.ndarray:
        arr = self._poly_cache.get(zone.zone_id)
        if arr is None:
            arr = np.array(zone.polygon, dtype=np.float32)
            self._poly_cache[zone.zone_id] = arr
        return arr

    def check(
        self,
        person_id: int,
        feet_points: list[tuple[float, float]],
    ) -> list[ZoneResult]:
        results: list[ZoneResult] = []
        for zone in self.manager.get_all():
            polygon = self._polygon(zone)

            inside = any(
                cv2.pointPolygonTest(polygon, (float(p[0]), float(p[1])), False) >= 0
                for p in feet_points
            )

            if inside:
                results.append(ZoneResult(
                    person_id=person_id,
                    zone_id=zone.zone_id,
                    zone_name=zone.name,
                    status=ZoneStatus.INSIDE,
                    distance_px=0.0,
                    severity=zone.severity,
                    required_ppe=zone.required_ppe,
                    delimita_avaliacao=zone.delimita_avaliacao,
                ))
                continue

            min_dist = self._min_distance(feet_points, polygon)
            if min_dist < self.buffer:
                results.append(ZoneResult(
                    person_id=person_id,
                    zone_id=zone.zone_id,
                    zone_name=zone.name,
                    status=ZoneStatus.APPROACHING,
                    distance_px=min_dist,
                    severity=zone.severity,
                    required_ppe=zone.required_ppe,
                    delimita_avaliacao=zone.delimita_avaliacao,
                ))

        return results

    @staticmethod
    def avaliacao_habilitada(zone_results: list[ZoneResult], all_zones: list[Zone]) -> bool:
        """Decide se EPI/pose devem ser avaliados para esta pessoa neste frame.

        Achado de campo (SPI 2026-09-16): zona so influenciava severidade de
        alerta, nunca o que era avaliado. Zonas com ``delimita_avaliacao=True``
        sao a area de trabalho declarada: se a camera tem ao menos uma, so
        quem esta dentro de alguma delas e avaliado. Zonas de perigo simples
        (delimita_avaliacao=False, o default) nunca restringem nada — so
        elevam severidade quando alguem entra nelas, como sempre fizeram. Sem
        NENHUMA zona delimitadora cadastrada, avalia todo mundo: comportamento
        anterior preservado pra nao regredir camera sem zona de trabalho
        configurada.
        """
        se_ha_delimitadora = any(z.delimita_avaliacao for z in all_zones)
        if not se_ha_delimitadora:
            return True
        return any(
            zr.status == ZoneStatus.INSIDE and zr.delimita_avaliacao
            for zr in zone_results
        )

    @staticmethod
    def _min_distance(points: list[tuple], polygon: np.ndarray) -> float:
        if not points:
            return float("inf")
        dists = [
            abs(cv2.pointPolygonTest(polygon, (float(p[0]), float(p[1])), True))
            for p in points
        ]
        return min(dists)
