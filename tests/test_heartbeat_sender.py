"""
Testes de HeartbeatSender (src/notify/heartbeat.py).

O que está sendo protegido: o throttle (no máximo um POST por intervalo) e a
tolerância a falha (API fora do ar não trava o pipeline nem faz retry a cada
frame) — design doc de gamificação E5, §2.1/§9 "Python".
"""
from __future__ import annotations

import pytest

from src.notify.heartbeat import HeartbeatSender
from src.notify.sink import MissingApiKeyError


class TransporteFake:
    def __init__(self, erro=None):
        self.erro = erro
        self.chamadas: list[tuple[str, dict]] = []

    def post(self, url, headers=None):
        self.chamadas.append((url, headers))
        if self.erro is not None:
            raise self.erro


def relogio(inicio=0.0):
    estado = {"t": inicio}

    def agora():
        return estado["t"]

    def avancar(segundos):
        estado["t"] += segundos

    return agora, avancar


def test_primeiro_envio_acontece_imediatamente():
    transporte = TransporteFake()
    agora, _ = relogio()
    sender = HeartbeatSender("http://api/api/v1", "CAM-01", api_key="chave-x",
                             transport=transporte, now=agora)

    sender.enviar_se_necessario()

    assert transporte.chamadas == [("http://api/api/v1/cameras/CAM-01/heartbeat", {"X-Cypher-Key": "chave-x"})]


def test_segundo_envio_antes_do_intervalo_e_ignorado():
    transporte = TransporteFake()
    agora, avancar = relogio()
    sender = HeartbeatSender("http://api/api/v1", "CAM-01", api_key="chave-x",
                             interval_seconds=30.0, transport=transporte, now=agora)
    sender.enviar_se_necessario()

    avancar(10)
    sender.enviar_se_necessario()

    assert len(transporte.chamadas) == 1


def test_envio_depois_do_intervalo_acontece_de_novo():
    transporte = TransporteFake()
    agora, avancar = relogio()
    sender = HeartbeatSender("http://api/api/v1", "CAM-01", api_key="chave-x",
                             interval_seconds=30.0, transport=transporte, now=agora)
    sender.enviar_se_necessario()

    avancar(31)
    sender.enviar_se_necessario()

    assert len(transporte.chamadas) == 2


def test_falha_de_transporte_nao_propaga_e_respeita_o_throttle_seguinte():
    transporte = TransporteFake(erro=ConnectionError("api fora do ar"))
    agora, avancar = relogio()
    sender = HeartbeatSender("http://api/api/v1", "CAM-01", api_key="chave-x",
                             interval_seconds=30.0, transport=transporte, now=agora)

    sender.enviar_se_necessario()  # nao deve lancar

    avancar(5)
    sender.enviar_se_necessario()  # ainda dentro do intervalo: nao tenta de novo

    assert len(transporte.chamadas) == 1


def test_sem_chave_resolvida_nao_envia_nem_lanca():
    transporte = TransporteFake()
    agora, _ = relogio()
    sender = HeartbeatSender("http://api/api/v1", "CAM-01", transport=transporte, now=agora)

    sender.enviar_se_necessario()

    assert transporte.chamadas == []


def test_api_key_env_resolvido_via_env_injetado():
    transporte = TransporteFake()
    agora, _ = relogio()
    sender = HeartbeatSender("http://api/api/v1", "CAM-01",
                             api_key_env="CYPHER_KEY", env={"CYPHER_KEY": "k1"},
                             transport=transporte, now=agora)

    sender.enviar_se_necessario()

    assert transporte.chamadas == [
        ("http://api/api/v1/cameras/CAM-01/heartbeat", {"X-Cypher-Key": "k1"})]


def test_api_key_explicito_vence_api_key_env():
    transporte = TransporteFake()
    agora, _ = relogio()
    sender = HeartbeatSender("http://api/api/v1", "CAM-01",
                             api_key="explicita", api_key_env="CYPHER_KEY",
                             env={"CYPHER_KEY": "do-env"},
                             transport=transporte, now=agora)

    sender.enviar_se_necessario()

    assert transporte.chamadas == [
        ("http://api/api/v1/cameras/CAM-01/heartbeat", {"X-Cypher-Key": "explicita"})]


def test_api_key_env_ausente_levanta_no_construtor():
    with pytest.raises(MissingApiKeyError):
        HeartbeatSender("http://api/api/v1", "CAM-01",
                        api_key_env="FALTA", env={}, transport=TransporteFake())
