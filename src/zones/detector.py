from __future__ import annotations

from dataclasses import dataclass
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


APPROACH_BUFFER_PX = 80


class ZoneDetector:
    """Tests whether a person's feet are inside, approaching, or outside each zone."""

    def __init__(self, manager: ZoneManager, approach_buffer_px: float = APPROACH_BUFFER_PX) -> None:
        self.manager = manager
        self.buffer = approach_buffer_px

    def check(
        self,
        person_id: int,
        feet_points: list[tuple[float, float]],
    ) -> list[ZoneResult]:
        results: list[ZoneResult] = []
        for zone in self.manager.get_all():
            polygon = np.array(zone.polygon, dtype=np.float32)

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
                ))

        return results

    @staticmethod
    def _min_distance(points: list[tuple], polygon: np.ndarray) -> float:
        if not points:
            return float("inf")
        dists = [
            abs(cv2.pointPolygonTest(polygon, (float(p[0]), float(p[1])), True))
            for p in points
        ]
        return min(dists)
