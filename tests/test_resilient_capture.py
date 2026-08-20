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
