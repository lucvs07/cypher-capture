"""
Testes de ``apply_rotation`` (src/utils/video.py).

O que está sendo protegido: cameras montadas fisicamente viradas (instalacao
tipo corredor) entregam o frame bruto do jeito que o sensor esta apontado,
sem nenhum metadado de rotacao no RTSP — quem sabe o angulo certo e o
operador, nao o OpenCV. --rotate so faz sentido se a transformacao geometrica
estiver correta: 90/270 tem que trocar largura x altura, e o pixel que
estava no canto errado tem que ir parar no canto certo.
"""
from __future__ import annotations

import numpy as np

from src.utils.video import apply_rotation


def test_rotate_zero_retorna_o_mesmo_array():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)

    resultado = apply_rotation(frame, 0)

    assert resultado is frame


def test_rotate_90_troca_largura_e_altura():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)  # height=100, width=200

    resultado = apply_rotation(frame, 90)

    assert resultado.shape == (200, 100, 3)


def test_rotate_90_horario_move_pixel_do_canto_superior_esquerdo_para_o_superior_direito():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    frame[0, 0] = (255, 0, 0)  # canto superior-esquerdo, marcado

    resultado = apply_rotation(frame, 90)

    assert tuple(resultado[0, -1]) == (255, 0, 0)
    assert resultado.shape == (20, 10, 3)


def test_rotate_270_e_o_inverso_de_90():
    frame = np.random.randint(0, 255, (30, 40, 3), dtype=np.uint8)

    ida_e_volta = apply_rotation(apply_rotation(frame, 90), 270)

    assert np.array_equal(ida_e_volta, frame)


def test_rotate_180_nao_troca_dimensoes():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    frame[0, 0] = (255, 0, 0)

    resultado = apply_rotation(frame, 180)

    assert resultado.shape == (10, 20, 3)
    assert tuple(resultado[-1, -1]) == (255, 0, 0)
