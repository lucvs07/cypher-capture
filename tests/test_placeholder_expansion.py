"""
Testes da expansão de credencial (src/capture/config_client.py).

O que está sendo protegido: que a senha da câmera nunca precise existir no
cadastro, e que a ausência de uma variável de ambiente produza um erro que
nomeia o que falta — em vez de uma URL com "${" literal indo para o OpenCV.
"""
from __future__ import annotations

import pytest

from src.capture.config_client import MissingCredentialError, expand_placeholders


def test_expande_uma_variavel():
    resultado = expand_placeholders(
        "rtsp://${CAM01_USER}@10.0.0.7/stream", {"CAM01_USER": "cypher"})

    assert resultado == "rtsp://cypher@10.0.0.7/stream"


def test_expande_varias_variaveis():
    resultado = expand_placeholders(
        "rtsp://${U}:${P}@10.0.0.7/stream", {"U": "cypher", "P": "s3nh4"})

    assert resultado == "rtsp://cypher:s3nh4@10.0.0.7/stream"


def test_texto_sem_placeholder_passa_intacto():
    assert expand_placeholders("data/validation/cenario2.mp4", {}) \
        == "data/validation/cenario2.mp4"
    assert expand_placeholders("0", {}) == "0"


def test_nenhum_valor_e_inventado_quando_a_variavel_falta():
    with pytest.raises(MissingCredentialError) as erro:
        expand_placeholders("rtsp://${CAM01_USER}:${CAM01_PASS}@h/s", {})

    # Todas de uma vez: descobrir uma por execução é hostil.
    assert erro.value.variaveis == ["CAM01_USER", "CAM01_PASS"]
    assert "CAM01_USER" in str(erro.value)
    assert "CAM01_PASS" in str(erro.value)


def test_reporta_apenas_a_que_falta():
    with pytest.raises(MissingCredentialError) as erro:
        expand_placeholders("rtsp://${U}:${P}@h/s", {"U": "cypher"})

    assert erro.value.variaveis == ["P"]


def test_variavel_vazia_conta_como_ausente():
    # Uma senha vazia produziria uma URL sintaticamente válida e um erro de
    # autenticação obscuro na câmera.
    with pytest.raises(MissingCredentialError):
        expand_placeholders("rtsp://${U}@h/s", {"U": ""})


def test_variavel_repetida_e_reportada_uma_vez():
    with pytest.raises(MissingCredentialError) as erro:
        expand_placeholders("rtsp://${U}:${U}@h/s", {})

    assert erro.value.variaveis == ["U"]


def test_fonte_nula_ou_vazia_nao_explode():
    assert expand_placeholders(None, {}) is None
    assert expand_placeholders("", {}) == ""
