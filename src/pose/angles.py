from __future__ import annotations

import math
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
    # Fallen-posture cue (velocity-independent; sustains the "person on the
    # ground" state). vertical_extent = (lowest_foot_y - nose_y)/bbox_height:
    # large when upright (feet far below the head), ~0 when lying flat, negative
    # when inverted. A forward bend keeps the feet planted low, so it stays
    # large — that's what separates a toe-touch from a real fall.
    vertical_extent: Optional[float] = None
    hip_center: Optional[tuple] = None              # mid-hip (x, y) px — stable CoM for fall velocity


def vertical_from_roll(roll_deg: float) -> tuple[float, float]:
    """Vetor "vertical do mundo" em coordenadas de imagem, para uma camera com
    roll de `roll_deg` graus (rotacao lateral, nao gira o frame inteiro como
    apply_rotation — so corrige o calculo de angulo). roll=0 reproduz o eixo Y
    hard-coded que o codigo assumia antes (0, -1).

    Convencao de sinal NAO validada empiricamente ainda — se a correcao sair
    invertida em campo, use o valor negado (ex.: -12 em vez de 12).
    """
    theta = math.radians(roll_deg)
    return (math.sin(theta), -math.cos(theta))


def _angle_from_vertical(vec: tuple[float, float], vertical: tuple[float, float]) -> float:
    """Menor angulo entre `vec` e `vertical` (ou seu oposto), em graus [0, 90].
    Generaliza o antigo arctan2(abs(dx), abs(dy)) para uma vertical arbitraria:
    dobra o resultado sobre 90 do mesmo jeito que abs() fazia para vertical=(0,-1)."""
    vx, vy = vec
    ux, uy = vertical
    norm_v = math.hypot(vx, vy)
    norm_u = math.hypot(ux, uy)
    if norm_v < 1e-6 or norm_u < 1e-6:
        return 0.0
    cos_a = max(-1.0, min(1.0, (vx * ux + vy * uy) / (norm_v * norm_u)))
    ang = math.degrees(math.acos(cos_a))
    return min(ang, 180.0 - ang)


def _angle(p1: tuple[float, float], vertex: tuple[float, float], p2: tuple[float, float]) -> float:
    """Interior angle at *vertex* formed by the vectors vertex→p1 and vertex→p2, in degrees [0, 180]."""
    v1 = np.array(p1) - np.array(vertex)
    v2 = np.array(p2) - np.array(vertex)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0  # degenerate → treat as fully extended (safe default)
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _midpoint(
    p1: Optional[tuple[float, float]],
    p2: Optional[tuple[float, float]],
) -> Optional[tuple[float, float]]:
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def extract_angles(
    pose: PersonPose,
    vertical_reference: tuple[float, float] = (0.0, -1.0),
) -> JointAngles:
    """Compute all joint angles from a PersonPose. Returns None for angles with low-confidence keypoints.

    `vertical_reference` e a direcao de "para cima" no mundo real, em
    coordenadas de imagem — default (0, -1) assume camera sem roll. Passe
    `vertical_from_roll(angulo)` para corrigir camera fisicamente inclinada.
    """
    g = pose.kp
    angles = JointAngles(person_id=pose.person_id)

    mid_shoulder = _midpoint(g("left_shoulder"), g("right_shoulder"))
    mid_hip = _midpoint(g("left_hip"), g("right_hip"))

    # Trunk flexion: angle between the torso vector (mid_hip → mid_shoulder)
    # and the world-vertical reference (image axis by default, or corrected
    # for camera roll). 0° = upright, 90° = horizontal torso.
    if mid_shoulder and mid_hip:
        dx = mid_shoulder[0] - mid_hip[0]
        dy = mid_shoulder[1] - mid_hip[1]  # image y grows downward
        angles.trunk_flexion = _angle_from_vertical((dx, dy), vertical_reference)

    # Neck: angle at mid-shoulder between nose and mid-hip (deviation from spine axis)
    nose = g("nose")
    if nose and mid_shoulder and mid_hip:
        angles.neck_flexion = _angle(nose, mid_shoulder, mid_hip)

    # Shoulder elevation: angle at shoulder formed by hip→shoulder→elbow
    # ~0° = arm hanging at side, ~90° = arm horizontal, ~180° = arm overhead
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

    # Stable centre-of-mass for fall velocity: the hip centre does NOT move when
    # the arms are raised/lowered (the bbox centre does), so using it kills the
    # arm-articulation false positives in TemporalAnalyzer.fall_risk_score.
    angles.hip_center = mid_hip or mid_shoulder

    # Vertical extent: distance from the nose down to the lowest visible lower-
    # body point (ankle, else knee, else hip), normalised by bbox height. Image
    # y grows downward, so upright = large positive; lying flat ≈ 0; inverted < 0.
    lower_pts = [
        g(n) for n in ("left_ankle", "right_ankle", "left_knee", "right_knee",
                       "left_hip", "right_hip")
    ]
    lower_pts = [p for p in lower_pts if p is not None]
    if nose and lower_pts:
        lowest_y = max(p[1] for p in lower_pts)
        angles.vertical_extent = (lowest_y - nose[1]) / pose.bbox_height

    return angles
