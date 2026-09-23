"""
Temporal analysis module — sliding-window metrics per tracked person.

Detects:
  - Movement repetitivity (autocorrelation of trunk angle)
  - Sudden acceleration (fall precursor)
  - Cumulative bad-posture time
  - Fall risk score (heuristic, 0–1)
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from .angles import JointAngles


@dataclass
class TemporalMetrics:
    person_id: int
    velocity_magnitude: float = 0.0       # pixels/frame of centre-of-mass
    acceleration_magnitude: float = 0.0   # change in velocity
    repetitivity_score: float = 0.0       # 0–1; high = cyclic motion (fatigue risk)
    sustained_bad_posture_s: float = 0.0  # seconds of continuously bad posture
    fall_risk_score: float = 0.0          # 0–1 heuristic (downward velocity + acceleration)


class TemporalAnalyzer:
    BUFFER_FRAMES = 60  # ~4 s at 15 fps

    def __init__(self, fps: float = 15.0) -> None:
        self.fps = fps
        self._com: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.BUFFER_FRAMES))
        self._trunk_angles: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.BUFFER_FRAMES))
        self._bad_posture_streak: dict[int, int] = defaultdict(int)

    def update(
        self,
        person_id: int,
        angles: JointAngles,
        bbox: tuple,
        is_bad_posture: bool,
    ) -> TemporalMetrics:
        # Centre of mass for velocity/fall-risk: prefer the hip centre (stable
        # under arm articulation) and fall back to the bbox centre only when the
        # hips are not visible. Using the bbox centre directly made raising or
        # lowering the arms — which resizes the bbox — read as vertical motion
        # and fire spurious fall_velocity alarms.
        if angles.hip_center is not None:
            com = angles.hip_center
        else:
            x1, y1, x2, y2 = bbox
            com = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self._com[person_id].append(com)
        # Skip occluded frames entirely — recording 0.0 for an undetermined
        # angle would inject a square-wave artifact into the autocorrelation.
        if angles.trunk_flexion is not None:
            self._trunk_angles[person_id].append(angles.trunk_flexion)

        if is_bad_posture:
            self._bad_posture_streak[person_id] += 1
        else:
            # Decay slowly so brief good frames don't reset the counter
            self._bad_posture_streak[person_id] = max(
                0, self._bad_posture_streak[person_id] - 1
            )

        metrics = TemporalMetrics(person_id=person_id)
        metrics.sustained_bad_posture_s = self._bad_posture_streak[person_id] / self.fps

        com_buf = np.array(self._com[person_id])
        if len(com_buf) >= 2:
            vels = np.linalg.norm(np.diff(com_buf, axis=0), axis=1)
            metrics.velocity_magnitude = float(vels[-1])
            if len(vels) >= 2:
                metrics.acceleration_magnitude = float(abs(vels[-1] - vels[-2]))

        angle_buf = np.array(self._trunk_angles[person_id])
        if len(angle_buf) >= 10:
            metrics.repetitivity_score = self._autocorrelation_score(angle_buf)

        # Fall risk: high downward velocity + high acceleration
        if len(com_buf) >= 3:
            # Positive y = downward in image coordinates
            dy = float(com_buf[-1][1] - com_buf[-3][1])
            metrics.fall_risk_score = min(
                1.0,
                max(0.0, metrics.velocity_magnitude * max(0.0, dy) / 300.0)
                + metrics.acceleration_magnitude / 50.0,
            )

        return metrics

    def forget(self, person_id: int) -> None:
        """Libera o estado de um track que saiu de quadro de vez."""
        self._com.pop(person_id, None)
        self._trunk_angles.pop(person_id, None)
        self._bad_posture_streak.pop(person_id, None)

    def get_history(self, person_id: int) -> dict:
        """Return raw buffers for time-series visualisation."""
        return {
            "com": list(self._com[person_id]),
            "trunk_flexion": list(self._trunk_angles[person_id]),
        }

    @staticmethod
    def _autocorrelation_score(signal: np.ndarray) -> float:
        """Normalised secondary autocorrelation peak as a repetitivity proxy."""
        signal = signal - signal.mean()
        std = signal.std()
        if std < 1e-6:
            return 0.0
        n = len(signal)
        corr = np.correlate(signal, signal, mode="full")[n - 1:]
        corr /= corr[0]
        # Small lags of any smooth signal correlate near 1.0 regardless of
        # periodicity, so the secondary peak is only meaningful after the
        # first zero crossing. No zero crossing in the window = no completed
        # cycle = not repetitive.
        negative = np.where(corr < 0)[0]
        if len(negative) == 0:
            return 0.0
        after = corr[negative[0]:]
        return float(max(0.0, np.max(after)))
