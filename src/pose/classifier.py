"""
Rule-based postural classifier with temporal hysteresis.

Taxonomy:
  NORMAL                   — no ergonomic concern detected
  ERGONOMICALLY_INADEQUATE — sustained bad posture per NR-17 thresholds
  IMMEDIATE_RISK           — fall or collapse detected (recall-optimised)
  CRITICAL_ZONE            — person inside a demarcated restricted area
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .angles import JointAngles
from .reba import REBAScore


class PostureCategory(Enum):
    NORMAL = "normal"
    ERGONOMICALLY_INADEQUATE = "ergonomically_inadequate"
    IMMEDIATE_RISK = "immediate_risk"
    CRITICAL_ZONE = "critical_zone"


@dataclass
class ClassifierThresholds:
    """All rule thresholds; defaults match docs/taxonomy.md."""
    trunk_flexion_deg: float = 60.0      # forward lean beyond this = spine load risk
    spine_load_knee_deg: float = 150.0   # knee angle above this counts as "straight leg"
    deep_squat_knee_deg: float = 90.0    # knee interior angle below this = deep squat
    elevated_arm_deg: float = 90.0       # shoulder interior angle below this = arm raised
    trunk_rotation_max: float = 0.35     # normalised shoulder/hip offset above this = instability
    fallen_aspect_ratio: float = 1.5     # bbox w/h above this = person horizontal
    reba_risk_score: int = 8             # REBA final score at/above this = ergonomic risk
    fall_risk_min: float = 0.7           # TemporalAnalyzer fall_risk_score threshold


@dataclass
class PostureState:
    category: PostureCategory = PostureCategory.NORMAL
    reason: str = ""
    frames_in_state: int = 0
    sustained_seconds: float = 0.0


class PostureClassifier:
    """
    Per-person state machine with hysteresis.

    Hysteresis design: entering a high-severity state is fast (enter_frames),
    exiting requires more evidence (exit_frames). This trades a small increase
    in false positives for the recall guarantee required on IMMEDIATE_RISK.
    """

    def __init__(
        self,
        fps: float = 15.0,
        thresholds: Optional[ClassifierThresholds] = None,
        enter_frames: int = 5,
        exit_frames: int = 15,
        enter_frames_risk: int = 2,
    ) -> None:
        self.fps = fps
        self.t = thresholds or ClassifierThresholds()
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        # IMMEDIATE_RISK uses a lower entry threshold to maximise recall
        self.enter_frames_risk = enter_frames_risk
        self._states: dict[int, PostureState] = {}
        self._pending: dict[int, tuple[PostureCategory, str, int]] = {}
        self._exit_counters: dict[int, int] = defaultdict(int)

    def update(
        self,
        person_id: int,
        angles: JointAngles,
        reba: REBAScore,
        bbox: tuple,
        in_critical_zone: bool = False,
        fall_risk: float = 0.0,
    ) -> PostureState:
        raw_cat, raw_reason = self._evaluate_rules(
            angles, reba, bbox, in_critical_zone, fall_risk
        )
        state = self._apply_hysteresis(person_id, raw_cat, raw_reason)

        if person_id not in self._states:
            self._states[person_id] = PostureState()

        current = self._states[person_id]
        if state is not None:
            current = state
            self._states[person_id] = current
            self._exit_counters[person_id] = 0
        elif current.category != PostureCategory.NORMAL:
            if raw_cat == PostureCategory.NORMAL:
                self._exit_counters[person_id] += 1
                if self._exit_counters[person_id] >= self.exit_frames:
                    current = PostureState()
                    self._states[person_id] = current
                    self._exit_counters[person_id] = 0
            else:
                # exit requires *consecutive* normal frames
                self._exit_counters[person_id] = 0

        current.frames_in_state += 1
        current.sustained_seconds = current.frames_in_state / self.fps
        return current

    def get_state(self, person_id: int) -> PostureState:
        return self._states.get(person_id, PostureState())

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------

    def _evaluate_rules(
        self,
        angles: JointAngles,
        reba: REBAScore,
        bbox: tuple,
        in_critical_zone: bool,
        fall_risk: float,
    ) -> tuple[PostureCategory, str]:
        # Priority order: IMMEDIATE_RISK > CRITICAL_ZONE > ERGONOMICALLY_INADEQUATE

        if self._detect_fallen(bbox):
            return PostureCategory.IMMEDIATE_RISK, "person_fallen"

        # Covers falls toward/away from the camera where the bbox never widens
        if fall_risk >= self.t.fall_risk_min:
            return PostureCategory.IMMEDIATE_RISK, "fall_velocity"

        if self._detect_instability(angles):
            return PostureCategory.IMMEDIATE_RISK, "balance_loss_risk"

        if in_critical_zone:
            return PostureCategory.CRITICAL_ZONE, "critical_zone_entry"

        if self._detect_spine_load(angles):
            return PostureCategory.ERGONOMICALLY_INADEQUATE, "trunk_flexion_spine_load"

        if self._detect_prolonged_squat(angles):
            return PostureCategory.ERGONOMICALLY_INADEQUATE, "prolonged_squat"

        if self._detect_elevated_arm(angles):
            return PostureCategory.ERGONOMICALLY_INADEQUATE, "elevated_arm_work"

        if reba.final_score >= self.t.reba_risk_score:
            return PostureCategory.ERGONOMICALLY_INADEQUATE, f"reba_{reba.final_score}"

        return PostureCategory.NORMAL, ""

    def _detect_fallen(self, bbox: tuple) -> bool:
        """Bbox significantly wider than tall → person horizontal (fallen)."""
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        return (w / h) > self.t.fallen_aspect_ratio

    def _detect_instability(self, angles: JointAngles) -> bool:
        """Trunk rotation proxy above threshold suggests rapid unexpected twist."""
        return (
            angles.trunk_rotation_proxy is not None
            and angles.trunk_rotation_proxy > self.t.trunk_rotation_max
        )

    def _detect_spine_load(self, angles: JointAngles) -> bool:
        """Trunk flexion > threshold with legs mostly straight → lifting with bent back (NR-17)."""
        if angles.trunk_flexion is None:
            return False
        if angles.trunk_flexion <= self.t.trunk_flexion_deg:
            return False
        knees = [v for v in [angles.left_knee_flexion, angles.right_knee_flexion] if v is not None]
        if not knees:
            return False
        # knee_flexion angle close to 180° means leg is straight
        return max(knees) > self.t.spine_load_knee_deg

    def _detect_prolonged_squat(self, angles: JointAngles) -> bool:
        """Knee interior angle below threshold → deep squat."""
        knees = [v for v in [angles.left_knee_flexion, angles.right_knee_flexion] if v is not None]
        if not knees:
            return False
        return min(knees) < self.t.deep_squat_knee_deg

    def _detect_elevated_arm(self, angles: JointAngles) -> bool:
        """Upper arm above shoulder level for sustained work (NR-17 risk factor)."""
        elevations = [
            v for v in [angles.left_shoulder_elevation, angles.right_shoulder_elevation]
            if v is not None
        ]
        if not elevations:
            return False
        # shoulder_elevation is angle hip→shoulder→elbow; below threshold = arm raised
        return any(e < self.t.elevated_arm_deg for e in elevations)

    # ------------------------------------------------------------------
    # Hysteresis
    # ------------------------------------------------------------------

    def _apply_hysteresis(
        self,
        person_id: int,
        cat: PostureCategory,
        reason: str,
    ) -> Optional[PostureState]:
        if cat == PostureCategory.NORMAL:
            # NORMAL is never "entered" through this path — leaving an alarm
            # state goes exclusively through the exit_frames counter in
            # update(), keeping alarms sticky (recall priority).
            self._pending.pop(person_id, None)
            return None

        pending = self._pending.get(person_id)
        enter_threshold = (
            self.enter_frames_risk
            if cat == PostureCategory.IMMEDIATE_RISK
            else self.enter_frames
        )

        if pending is None or pending[0] != cat:
            self._pending[person_id] = (cat, reason, 1)
            return None

        count = pending[2] + 1
        self._pending[person_id] = (cat, reason, count)

        current_cat = self._states.get(person_id, PostureState()).category
        if cat != current_cat and count >= enter_threshold:
            self._pending.pop(person_id)
            # The posture already persisted `count` raw frames before being
            # confirmed; start the counter there so sustained_seconds is honest
            # (update() increments once more right after).
            return PostureState(category=cat, reason=reason, frames_in_state=count - 1)

        return None
