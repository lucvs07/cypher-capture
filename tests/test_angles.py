"""
Testes de src/pose/angles.py - foco na generalizacao de trunk_flexion para
aceitar uma referencia de verticalidade (correcao de roll de camera), que
precisa reproduzir o comportamento antigo (assumir roll=0) quando ninguem
passa vertical_reference.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.pose.angles import extract_angles, vertical_from_roll
from src.pose.estimator import KP, PersonPose


def make_pose(mid_shoulder: tuple[float, float], mid_hip: tuple[float, float]) -> PersonPose:
    """Pose so com ombro/quadril, confianca alta, na posicao dada (o resto fica ausente)."""
    keypoints = np.zeros((17, 2))
    confidences = np.zeros(17)
    for side in ("left", "right"):
        keypoints[KP[f"{side}_shoulder"]] = mid_shoulder
        confidences[KP[f"{side}_shoulder"]] = 1.0
        keypoints[KP[f"{side}_hip"]] = mid_hip
        confidences[KP[f"{side}_hip"]] = 1.0
    return PersonPose(person_id=1, keypoints=keypoints, confidences=confidences,
                       bbox=(0.0, 0.0, 100.0, 200.0))


def test_vertical_from_roll_zero_e_eixo_y_negativo():
    assert vertical_from_roll(0.0) == pytest.approx((0.0, -1.0))


def test_vertical_from_roll_noventa_graus():
    vx, vy = vertical_from_roll(90.0)
    assert (vx, vy) == pytest.approx((1.0, 0.0), abs=1e-9)


def test_trunk_flexion_upright_sem_correcao():
    """Pessoa em pe (ombro acima do quadril), sem roll: flexao ~0."""
    pose = make_pose(mid_shoulder=(50.0, 50.0), mid_hip=(50.0, 150.0))
    angles = extract_angles(pose)
    assert angles.trunk_flexion == pytest.approx(0.0, abs=1e-6)


def test_trunk_flexion_default_bate_com_formula_antiga():
    """dx=30, dy=-100 (ombro acima e a direita do quadril) - reproduz
    arctan2(abs(dx), abs(dy)) da implementacao anterior."""
    pose = make_pose(mid_shoulder=(80.0, 50.0), mid_hip=(50.0, 150.0))
    angles = extract_angles(pose)
    esperado = math.degrees(math.atan2(30.0, 100.0))
    assert angles.trunk_flexion == pytest.approx(esperado, abs=1e-6)


def test_trunk_flexion_corrigida_por_roll_da_camera():
    """Pessoa fisicamente em pe, mas a camera tem roll de 30 graus: sem
    correcao o torso pareceria inclinado 30 graus; com a referencia vertical
    rotacionada, a flexao medida volta a ~0."""
    theta = math.radians(30.0)
    comprimento = 100.0
    dx = comprimento * math.sin(theta)
    dy = -comprimento * math.cos(theta)
    mid_hip = (50.0, 150.0)
    mid_shoulder = (mid_hip[0] + dx, mid_hip[1] + dy)

    pose = make_pose(mid_shoulder=mid_shoulder, mid_hip=mid_hip)
    vertical = vertical_from_roll(30.0)

    angles = extract_angles(pose, vertical_reference=vertical)
    assert angles.trunk_flexion == pytest.approx(0.0, abs=1e-6)
