"""
Testes da configuração de captura.

O que está sendo protegido: que o transporte RTSP seja TCP por padrão (UDP
sob carga gera artefato que o detector lê como movimento), que as duas grafias
da chave de timeout sejam declaradas, e que o pipeline continue rodando como
antes para quem não configurou nada.
"""
from __future__ import annotations

import textwrap

from src.capture import CaptureConfig, aplicar_opcoes_ffmpeg


def test_valores_padrao(tmp_path):
    config = tmp_path / "vazio.yaml"
    config.write_text("pipeline:\n  device: cpu\n")

    capture = CaptureConfig.from_yaml(str(config))

    assert capture.rtsp_transport == "tcp"
    assert capture.read_timeout_ms == 5000
    assert capture.backoff_base_seconds == 0.5
    assert capture.backoff_max_seconds == 10.0
    assert capture.cache_path == "outputs/camera-config-cache.json"


def test_le_a_secao_capture(tmp_path):
    config = tmp_path / "cheio.yaml"
    config.write_text(textwrap.dedent("""
        capture:
          rtsp_transport: udp
          read_timeout_ms: 2000
          backoff_base_seconds: 1
          backoff_max_seconds: 30
          cache_path: outputs/meu-cache.json
    """))

    capture = CaptureConfig.from_yaml(str(config))

    assert capture.rtsp_transport == "udp"
    assert capture.read_timeout_ms == 2000
    assert capture.backoff_base_seconds == 1
    assert capture.backoff_max_seconds == 30
    assert capture.cache_path == "outputs/meu-cache.json"


def test_chaves_desconhecidas_sao_ignoradas(tmp_path):
    config = tmp_path / "extra.yaml"
    config.write_text("capture:\n  rtsp_transport: tcp\n  invento: 42\n")

    assert CaptureConfig.from_yaml(str(config)).rtsp_transport == "tcp"


def test_declara_as_duas_grafias_da_chave_de_timeout():
    env: dict[str, str] = {}

    aplicar_opcoes_ffmpeg(CaptureConfig(), env)

    opcoes = env["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
    assert "rtsp_transport;tcp" in opcoes
    # stimeout no FFmpeg antigo, timeout no recente. Declarar só uma faz a
    # protecao nao existir na outra versao, sem avisar.
    assert "stimeout;5000000" in opcoes
    assert "timeout;5000000" in opcoes


def test_timeout_e_convertido_para_microssegundos():
    env: dict[str, str] = {}

    aplicar_opcoes_ffmpeg(CaptureConfig(read_timeout_ms=2000), env)

    assert "stimeout;2000000" in env["OPENCV_FFMPEG_CAPTURE_OPTIONS"]


def test_nao_sobrescreve_opcoes_definidas_pelo_operador():
    env = {"OPENCV_FFMPEG_CAPTURE_OPTIONS": "rtsp_transport;udp"}

    aplicar_opcoes_ffmpeg(CaptureConfig(), env)

    assert env["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;udp"
