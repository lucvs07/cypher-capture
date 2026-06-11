from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .estimator import PersonPose


@dataclass
class JointAngles:
    person_id: int
    trunk_flexion: Optional[float] = None           # degrees, forward lean of torso
    neck_flexion: Optional[float] = None            # degrees from vertical
    left_shoulder_elevation: Optional[float] = None  # angle hip→shoulder→elbow
    right_shoulder_elevation: Optional[float] = None
    left_elbow_flexion: Optional[float] = None      # angle shoulder→elbow→wrist
    right_elbow_flexion: Optional[float] = None
    left_knee_flexion: Optional[float] = None       # angle hip→knee→ankle
    right_knee_flexion: Optional[float] = None
    trunk_rotation_proxy: Optional[float] = None    # shoulder/hip horizontal offset, normalised by height


def _angle(p1: tuple[float, float], vertex: tuple[float, float], p2: tuple[float, float]) -> float:
    """Interior angle at *vertex* formed by the vectors vertex→p1 and vertex→p2, in degrees [0, 180]."""
    v1 = np.array(p1) - np.array(vertex)
    v2 = np.array(p2) - np.array(vertex)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _midpoint(
    p1: Optional[tuple[float, float]],
    p2: Optional[tuple[float, float]],
) -> Optional[tuple[float, float]]:
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def extract_angles(pose: PersonPose) -> JointAngles:
    """Compute all joint angles from a PersonPose. Returns None for angles with low-confidence keypoints."""
    g = pose.kp
    angles = JointAngles(person_id=pose.person_id)

    mid_shoulder = _midpoint(g("left_shoulder"), g("right_shoulder"))
    mid_hip = _midpoint(g("left_hip"), g("right_hip"))
    mid_knee = _midpoint(g("left_knee"), g("right_knee"))

    # Trunk flexion: angle at hip between mid-shoulder and mid-knee
    # 0° = upright, 90° = fully bent forward
    if mid_shoulder and mid_hip and mid_knee:
        angles.trunk_flexion = 180.0 - _angle(mid_shoulder, mid_hip, mid_knee)

    # Neck: angle at mid-shoulder between nose and mid-hip (deviation from spine axis)
    nose = g("nose")
    if nose and mid_shoulder and mid_hip:
        angles.neck_flexion = _angle(nose, mid_shoulder, mid_hip)

    # Shoulder elevation: angle at shoulder formed by hip→shoulder→elbow
    # 180° = arm hanging, ~90° = arm horizontal, ~0° = arm overhead
    for side in ("left", "right"):
        hip = g(f"{side}_hip")
        shoulder = g(f"{side}_shoulder")
        elbow = g(f"{side}_elbow")
        if hip and shoulder and elbow:
            setattr(angles, f"{side}_shoulder_elevation", _angle(hip, shoulder, elbow))

    # Elbow flexion: angle at elbow formed by shoulder→elbow→wrist
    for side in ("left", "right"):
        shoulder = g(f"{side}_shoulder")
        elbow = g(f"{side}_elbow")
        wrist = g(f"{side}_wrist")
        if shoulder and elbow and wrist:
            setattr(angles, f"{side}_elbow_flexion", _angle(shoulder, elbow, wrist))

    # Knee flexion: angle at knee formed by hip→knee→ankle
    for side in ("left", "right"):
        hip = g(f"{side}_hip")
        knee = g(f"{side}_knee")
        ankle = g(f"{side}_ankle")
        if hip and knee and ankle:
            setattr(angles, f"{side}_knee_flexion", _angle(hip, knee, ankle))

    # Trunk rotation proxy: normalised horizontal offset between mid-shoulder and mid-hip
    # Non-zero means asymmetric shoulder/hip alignment (2D approximation of axial rotation)
    if mid_shoulder and mid_hip:
        angles.trunk_rotation_proxy = abs(mid_shoulder[0] - mid_hip[0]) / pose.bbox_height

    return angles
