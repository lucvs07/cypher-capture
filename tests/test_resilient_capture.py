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


import threading
import time


class CapturaAoVivoFake:
    """Câmera controlável: entrega frames sob comando, falha sob comando."""

    def __init__(self, abre=True):
        self.abre = abre
        self.liberada = False
        self.falhar = False
        self.travar = threading.Event()
        self.entregues = 0
        self.props: dict[int, float] = {}

    def isOpened(self):  # noqa: N802
        return self.abre

    def read(self):
        if self.travar.is_set():
            # Câmera conectada mas muda: read() nunca retorna.
            time.sleep(0.05)
            return False, None
        if self.falhar:
            return False, None
        self.entregues += 1
        return True, np.full((48, 64, 3), self.entregues % 255, dtype=np.uint8)

    def release(self):
        self.liberada = True

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 64
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 48
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def set(self, prop, value):
        self.props[prop] = value
        return True


def _esperar(condicao, timeout=3.0):
    limite = time.perf_counter() + timeout
    while time.perf_counter() < limite:
        if condicao():
            return True
        time.sleep(0.01)
    return False


def test_ao_vivo_le_numa_thread_e_nao_bloqueia_o_laco():
    fake = CapturaAoVivoFake()
    cap = ResilientCapture("rtsp://10.0.0.7/stream", factory=lambda: fake)
    cap.open()
    try:
        assert _esperar(lambda: fake.entregues > 0)

        inicio = time.perf_counter()
        for _ in range(50):
            cap.read()
        decorrido = time.perf_counter() - inicio

        # 50 leituras do laço quente: nenhuma pode esperar pela câmera.
        assert decorrido < 0.05
    finally:
        cap.release()


def test_ao_vivo_pede_buffer_minimo_ao_backend():
    fake = CapturaAoVivoFake()
    cap = ResilientCapture("rtsp://10.0.0.7/stream", factory=lambda: fake)
    cap.open()
    try:
        import cv2
        # Best-effort: o FFmpeg costuma ignorar, e é por isso que a thread
        # existe. Ainda assim pedimos.
        assert fake.props.get(cv2.CAP_PROP_BUFFERSIZE) == 1
    finally:
        cap.release()


def test_ao_vivo_entrega_o_frame_mais_recente_e_descarta_os_atrasados():
    fake = CapturaAoVivoFake()
    cap = ResilientCapture("rtsp://10.0.0.7/stream", factory=lambda: fake)
    cap.open()
    try:
        assert _esperar(lambda: fake.entregues > 20)
        ok, _ = cap.read()

        assert ok is True
        # A câmera produziu mais do que o laço consumiu: a diferença foi
        # descartada em vez de virar latência.
        assert cap.frames_descartados > 0
    finally:
        cap.release()


def test_ao_vivo_sem_frame_novo_devolve_falso_sem_erro():
    fake = CapturaAoVivoFake()
    cap = ResilientCapture("rtsp://10.0.0.7/stream", factory=lambda: fake)
    cap.open()
    try:
        assert _esperar(lambda: cap.read()[0] is True)

        # Duas leituras seguidas sem a câmera ter produzido nada novo.
        fake.travar.set()
        time.sleep(0.1)
        cap.read()
        ok, frame = cap.read()

        assert ok is False
        assert frame is None
        # Travar não é morrer: o laço continua girando.
        assert cap.state in ("conectado", "reconectando")
    finally:
        fake.travar.clear()
        cap.release()


def test_ao_vivo_reconecta_quando_a_camera_cai():
    criadas: list[CapturaAoVivoFake] = []

    def fabrica():
        nova = CapturaAoVivoFake()
        criadas.append(nova)
        return nova

    cap = ResilientCapture("rtsp://10.0.0.7/stream", factory=fabrica,
                           backoff_base=0.01, backoff_max=0.05)
    cap.open()
    try:
        assert _esperar(lambda: criadas[0].entregues > 0)

        criadas[0].falhar = True

        assert _esperar(lambda: cap.reconnects >= 1)
        assert _esperar(lambda: len(criadas) >= 2)
        # A captura antiga foi liberada antes de reabrir.
        assert criadas[0].liberada is True
        # E o pipeline segue vivo.
        assert cap.isOpened() is True
    finally:
        cap.release()


def test_reconexao_nao_derruba_o_processo_quando_a_camera_nao_volta():
    def fabrica():
        return CapturaAoVivoFake(abre=False)

    cap = ResilientCapture("rtsp://10.0.0.7/stream", factory=fabrica,
                           backoff_base=0.01, backoff_max=0.02)
    # A primeira abertura falha e é erro do operador: fonte errada.
    with pytest.raises(RuntimeError):
        cap.open()


def test_release_encerra_a_thread():
    fake = CapturaAoVivoFake()
    cap = ResilientCapture("rtsp://10.0.0.7/stream", factory=lambda: fake)
    cap.open()
    assert _esperar(lambda: fake.entregues > 0)

    cap.release()

    assert _esperar(lambda: fake.liberada is True)
    assert cap.isOpened() is False
