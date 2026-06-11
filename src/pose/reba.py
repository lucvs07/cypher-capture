"""
REBA (Rapid Entire Body Assessment) scoring engine.
Reference: Hignett & McAtamney, Applied Ergonomics, 31(2), 2000.

2D vision limitations declared:
  - Wrist deviation/twist cannot be reliably estimated from monocular 2D; score fixed at 1.
  - Axial trunk rotation is approximated via shoulder/hip horizontal offset.
  - Load/force and coupling adjustments require domain knowledge; conservative defaults assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .angles import JointAngles

# ---------------------------------------------------------------------------
# REBA scoring tables
# ---------------------------------------------------------------------------

# Table A: [trunk_score-1][neck_score-1][legs_score-1] → score (1–9)
# trunk: 1–5, neck: 1–3, legs: 1–4
_TABLE_A = [
    [[1, 2, 3, 4], [1, 2, 3, 4], [3, 3, 5, 6]],   # trunk=1
    [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],   # trunk=2
    [[2, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],   # trunk=3
    [[3, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],   # trunk=4
    [[4, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],   # trunk=5
]

# Table B: [upper_arm_score-1][lower_arm_score-1][wrist_score-1] → score (1–8)
# upper_arm: 1–6, lower_arm: 1–2, wrist: 1–3
_TABLE_B = [
    [[1, 2, 2], [1, 2, 3]],   # upper_arm=1
    [[1, 2, 3], [2, 3, 4]],   # upper_arm=2
    [[3, 4, 5], [3, 4, 5]],   # upper_arm=3
    [[4, 5, 5], [4, 5, 5]],   # upper_arm=4
    [[6, 7, 7], [6, 7, 7]],   # upper_arm=5
    [[7, 8, 8], [7, 8, 8]],   # upper_arm=6
]

# Table C: [score_a-1][score_b-1] → final_score (1–12)
_TABLE_C = [
    [ 1,  1,  1,  2,  3,  3,  4,  5,  6,  7,  7,  7],  # A=1
    [ 1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  7,  8],  # A=2
    [ 2,  3,  3,  3,  4,  5,  6,  7,  7,  8,  8,  8],  # A=3
    [ 3,  4,  4,  4,  5,  6,  7,  8,  8,  9,  9,  9],  # A=4
    [ 4,  4,  4,  5,  6,  7,  8,  8,  9,  9, 10, 10],  # A=5
    [ 6,  6,  6,  7,  8,  8,  9,  9, 10, 10, 11, 11],  # A=6
    [ 7,  7,  7,  8,  9,  9,  9, 10, 11, 11, 11, 12],  # A=7
    [ 8,  8,  8,  9, 10, 10, 10, 10, 11, 11, 12, 12],  # A=8
    [ 9,  9,  9,  9, 10, 10, 11, 11, 12, 12, 12, 12],  # A=9
    [10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 12],  # A=10
    [11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12],  # A=11
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],  # A=12
]

_ACTION_LEVELS = {
    1: "Negligible risk — no action required",
    2: "Low risk — change may be needed",
    3: "Medium risk — investigate and change soon",
    4: "High risk — investigate and change immediately",
}


@dataclass
class REBAScore:
    person_id: int
    trunk_score: int = 1
    neck_score: int = 1
    legs_score: int = 1
    upper_arm_score: int = 1
    lower_arm_score: int = 1
    wrist_score: int = 1
    score_a: int = 1
    score_b: int = 1
    final_score: int = 1
    action_level: int = 1
    action_description: str = "Negligible risk — no action required"


# ---------------------------------------------------------------------------
# Individual joint scoring functions
# ---------------------------------------------------------------------------

def _trunk_score(flexion: Optional[float]) -> int:
    if flexion is None:
        return 1
    if flexion <= 5:
        return 1
    if flexion <= 20:
        return 2
    if flexion <= 60:
        return 3
    return 4


def _neck_score(flexion: Optional[float]) -> int:
    """neck_flexion here is the angle nose→mid_shoulder→mid_hip; ~180° = upright, decreases when bent."""
    if flexion is None:
        return 1
    # Convert to deviation from upright: 0° = perfectly upright line
    deviation = abs(180.0 - flexion)
    if deviation <= 20:
        return 1
    return 2


def _legs_score(left_knee: Optional[float], right_knee: Optional[float]) -> int:
    """Bilateral standing assumed; score driven by worst knee flexion."""
    knees = [k for k in [left_knee, right_knee] if k is not None]
    if not knees:
        return 1
    # knee_flexion angle: ~180° = straight leg, decreases with more bend
    # Convert: bend_deg = 180 - knee_flexion
    worst_bend = max(180.0 - k for k in knees)
    if worst_bend <= 30:
        return 1
    if worst_bend <= 60:
        return 2
    return 3


def _upper_arm_score(left_elev: Optional[float], right_elev: Optional[float]) -> int:
    """
    shoulder_elevation = angle hip→shoulder→elbow.
    180° ≈ arm hanging, ~90° ≈ arm horizontal, ~0° ≈ arm overhead.
    REBA scores deviation from vertical (0° from hang = 0° deviation).
    """
    elevations = [e for e in [left_elev, right_elev] if e is not None]
    if not elevations:
        return 1
    worst_deviation = max(abs(180.0 - e) for e in elevations)
    if worst_deviation <= 20:
        return 1
    if worst_deviation <= 45:
        return 2
    if worst_deviation <= 90:
        return 3
    if worst_deviation <= 110:
        return 4
    return 5


def _lower_arm_score(left_elbow: Optional[float], right_elbow: Optional[float]) -> int:
    """elbow_flexion: 60–100° is functional range → score 1, outside → score 2."""
    elbows = [e for e in [left_elbow, right_elbow] if e is not None]
    if not elbows:
        return 1
    worst = min(elbows)  # most extended/flexed
    if 60 <= worst <= 100:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_reba(angles: JointAngles) -> REBAScore:
    score = REBAScore(person_id=angles.person_id)

    score.trunk_score = _trunk_score(angles.trunk_flexion)
    score.neck_score = _neck_score(angles.neck_flexion)
    score.legs_score = _legs_score(angles.left_knee_flexion, angles.right_knee_flexion)
    score.upper_arm_score = _upper_arm_score(
        angles.left_shoulder_elevation, angles.right_shoulder_elevation
    )
    score.lower_arm_score = _lower_arm_score(
        angles.left_elbow_flexion, angles.right_elbow_flexion
    )
    score.wrist_score = 1  # 2D vision cannot reliably estimate wrist deviation

    # Table A lookup
    t_idx = min(score.trunk_score - 1, 4)
    n_idx = min(score.neck_score - 1, 2)
    l_idx = min(score.legs_score - 1, 3)
    score.score_a = _TABLE_A[t_idx][n_idx][l_idx]

    # Table B lookup
    ua_idx = min(score.upper_arm_score - 1, 5)
    la_idx = min(score.lower_arm_score - 1, 1)
    w_idx = min(score.wrist_score - 1, 2)
    score.score_b = _TABLE_B[ua_idx][la_idx][w_idx]

    # Table C
    a_idx = min(score.score_a - 1, 11)
    b_idx = min(score.score_b - 1, 11)
    score.final_score = _TABLE_C[a_idx][b_idx]

    # Action level
    if score.final_score == 1:
        score.action_level = 1
    elif score.final_score <= 3:
        score.action_level = 2
    elif score.final_score <= 7:
        score.action_level = 3
    else:
        score.action_level = 4

    score.action_description = _ACTION_LEVELS[score.action_level]
    return score
