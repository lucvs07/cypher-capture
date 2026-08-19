"""
Testes do servidor de vídeo ao vivo (src/notify/mjpeg.py).

O que está sendo protegido: que publicar um frame seja barato e que o servidor
não derrube o pipeline quando ninguém está assistindo — ou quando o browser
desconecta no meio do stream.
"""
from __future__ import annotations

import urllib.error
import urllib.request

import numpy as np
import pytest

from src.notify.mjpeg import MjpegServer


@pytest.fixture
def servidor():
    srv = MjpegServer(port=0)
    srv.start()
    yield srv
    srv.stop()


def frame_de_teste() -> np.ndarray:
    return np.zeros((48, 64, 3), dtype=np.uint8)


def test_sobe_numa_porta_efemera(servidor):
    assert servidor.port > 0


def test_comeca_sem_frame(servidor):
    assert servidor.has_frame() is False


def test_publicar_e_barato_e_nao_codifica(servidor):
    import time

    inicio = time.perf_counter()
    for _ in range(200):
        servidor.publish(frame_de_teste())
    decorrido = time.perf_counter() - inicio

    assert servidor.has_frame() is True
    # 200 publicações sem cliente conectado: nenhuma codificação JPEG.
    assert decorrido < 0.05


def test_responde_o_stream_como_multipart(servidor):
    servidor.publish(frame_de_teste())

    with urllib.request.urlopen(f"http://127.0.0.1:{servidor.port}/stream",
                                timeout=5) as resposta:
        tipo = resposta.headers["Content-Type"]
        pedaco = resposta.read(512)

    assert tipo.startswith("multipart/x-mixed-replace")
    assert b"Content-Type: image/jpeg" in pedaco
    assert b"\xff\xd8" in pedaco          # marcador de início de JPEG


def test_rota_desconhecida_devolve_404(servidor):
    with pytest.raises(urllib.error.HTTPError) as erro:
        urllib.request.urlopen(f"http://127.0.0.1:{servidor.port}/inexistente",
                               timeout=5)

    assert erro.value.code == 404
