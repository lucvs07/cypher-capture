"""
Testes da captura resiliente (src/capture/resilient.py).

O que está sendo protegido: que câmera IP que cai, trava ou entrega frames
mais rápido do que o pipeline processa não derrube nem atrase a análise — e
que arquivo continue sendo lido quadro a quadro, porque descartar frame de
arquivo mudaria o conteúdo analisado e invalidaria as métricas de E1.

Nenhum teste aqui abre câmera ou arquivo: a captura é injetada.
"""
from __future__ import annotations

import pytest

from src.utils.video import is_live_source


@pytest.mark.parametrize("fonte", [0, 1, "0", "-1", "rtsp://10.0.0.7/stream",
                                   "rtsps://cam/stream", "http://cam/mjpeg",
                                   "https://cam/mjpeg", "udp://239.0.0.1:1234"])
def test_reconhece_fonte_ao_vivo(fonte):
    assert is_live_source(fonte) is True


@pytest.mark.parametrize("fonte", ["data/validation/cenario2.mp4",
                                   "/tmp/clipe.avi", "clipe.mkv",
                                   "outputs/captures/x.mp4"])
def test_reconhece_arquivo(fonte):
    assert is_live_source(fonte) is False


def test_caminho_de_diretorio_de_imagens_nao_e_ao_vivo():
    assert is_live_source("data/imagens/") is False


import numpy as np

from src.capture.resilient import ResilientCapture


class CapturaFake:
    """Captura controlável: entrega N frames e depois falha ou encerra."""

    def __init__(self, frames=3, abre=True, largura=64, altura=48, fps=30.0):
        self.restantes = frames
        self.abre = abre
        self.largura = largura
        self.altura = altura
        self.fps = fps
        self.liberada = False
        self.props: dict[int, float] = {}

    def isOpened(self):  # noqa: N802 (assinatura do OpenCV)
        return self.abre

    def read(self):
        if self.restantes <= 0:
            return False, None
        self.restantes -= 1
        return True, np.zeros((self.altura, self.largura, 3), dtype=np.uint8)

    def release(self):
        self.liberada = True

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.largura
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.altura
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0

    def set(self, prop, value):
        self.props[prop] = value
        return True


def test_le_arquivo_quadro_a_quadro_sem_descartar():
    fake = CapturaFake(frames=3)
    cap = ResilientCapture("clipe.mp4", factory=lambda: fake)
    cap.open()

    lidos = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lidos.append(frame)

    assert len(lidos) == 3
    assert cap.frames_descartados == 0
    cap.release()


def test_fim_de_arquivo_encerra_em_vez_de_reconectar():
    fake = CapturaFake(frames=1)
    cap = ResilientCapture("clipe.mp4", factory=lambda: fake)
    cap.open()

    cap.read()
    ok, _ = cap.read()

    assert ok is False
    assert cap.state == "encerrado"
    assert cap.reconnects == 0
    cap.release()


def test_expoe_dimensoes_e_fps_da_fonte():
    cap = ResilientCapture("clipe.mp4",
                           factory=lambda: CapturaFake(largura=1920, altura=1080, fps=25.0))
    cap.open()

    assert cap.width == 1920
    assert cap.height == 1080
    assert cap.fps == pytest.approx(25.0)
    cap.release()


def test_dimensoes_ausentes_caem_para_um_default_utilizavel():
    cap = ResilientCapture("clipe.mp4",
                           factory=lambda: CapturaFake(largura=0, altura=0, fps=0.0))
    cap.open()

    # O writer de vídeo do run_pipeline precisa de números; zero o quebraria.
    assert cap.width == 1280
    assert cap.height == 720
    assert cap.fps > 0
    cap.release()


def test_arquivo_que_nao_abre_falha_imediatamente():
    cap = ResilientCapture("nao-existe.mp4", factory=lambda: CapturaFake(abre=False))

    with pytest.raises(RuntimeError, match="nao-existe.mp4"):
        cap.open()


def test_release_libera_a_captura_subjacente():
    fake = CapturaFake()
    cap = ResilientCapture("clipe.mp4", factory=lambda: fake)
    cap.open()

    cap.release()

    assert fake.liberada is True
    assert cap.isOpened() is False
