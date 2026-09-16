"""
Testes do cliente de config de câmera (src/capture/config_client.py).

O que está sendo protegido: a escada de degradação. Um sistema de segurança
não pode ficar impedido de capturar porque o servidor de configuração está
fora — mas também não pode subir apontando para uma câmera que alguém
desativou de propósito. Os dois casos parecem "erro de rede" e não são.

Nenhum teste aqui toca a rede: o transporte é injetado.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.capture.config_client import (
    CameraConfigClient,
    CameraDisabledError,
    CameraNotRegisteredError,
    MissingCredentialError,
)

AGORA = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)

CORPO = {
    "codigo": "CAM-01",
    "nome": "Esteira - frontal",
    "estacaoId": "EST-01",
    "fonte": "rtsp://${CAM01_USER}:${CAM01_PASS}@10.0.0.7/stream",
    "streamPort": 8090,
    "ativa": True,
}

AMBIENTE = {"CAM01_USER": "cypher", "CAM01_PASS": "s3nh4"}


class TransporteFake:
    """Responde um status fixo, ou levanta, e registra as URLs pedidas."""

    def __init__(self, status=200, corpo=None, erro=None):
        self.status = status
        self.corpo = corpo if corpo is not None else CORPO
        self.erro = erro
        self.urls: list[str] = []
        self.api_keys: list[object] = []

    def get_json(self, url, api_key=None):
        self.urls.append(url)
        self.api_keys.append(api_key)
        if self.erro is not None:
            raise self.erro
        return self.status, self.corpo


def cliente(tmp_path, transporte, **kwargs):
    return CameraConfigClient(
        "http://api/api/v1",
        cache_path=str(tmp_path / "cache.json"),
        transport=transporte,
        now=lambda: AGORA,
        **kwargs,
    )


def test_busca_no_endpoint_de_config_da_camera(tmp_path):
    transporte = TransporteFake()

    config = cliente(tmp_path, transporte).fetch("CAM-01", AMBIENTE)

    assert transporte.urls == ["http://api/api/v1/cameras/CAM-01/config"]
    assert config.codigo == "CAM-01"
    assert config.estacao_id == "EST-01"
    assert config.stream_port == 8090
    assert config.origem == "api"


def test_expande_a_credencial_do_ambiente(tmp_path):
    config = cliente(tmp_path, TransporteFake()).fetch("CAM-01", AMBIENTE)

    assert config.fonte == "rtsp://cypher:s3nh4@10.0.0.7/stream"


def test_rotacao_e_angulo_vem_da_api(tmp_path):
    corpo = {**CORPO, "rotacao": 90, "anguloCameraGraus": -12}
    transporte = TransporteFake(corpo=corpo)

    config = cliente(tmp_path, transporte).fetch("CAM-01", AMBIENTE)

    assert config.rotacao == 90
    assert config.angulo_camera_graus == -12


def test_rotacao_e_angulo_default_zero_quando_ausentes(tmp_path):
    """Backend ainda nao redeployado com os campos novos - nao pode quebrar."""
    transporte = TransporteFake(corpo=CORPO)  # CORPO nao tem rotacao/anguloCameraGraus

    config = cliente(tmp_path, transporte).fetch("CAM-01", AMBIENTE)

    assert config.rotacao == 0
    assert config.angulo_camera_graus == 0


def test_credencial_ausente_aborta_em_vez_de_montar_url_quebrada(tmp_path):
    with pytest.raises(MissingCredentialError):
        cliente(tmp_path, TransporteFake()).fetch("CAM-01", {})


def test_envia_x_cypher_key_quando_api_key_fornecida(tmp_path):
    transporte = TransporteFake()

    cliente(tmp_path, transporte).fetch("CAM-01", AMBIENTE, api_key="cam-key-abc")

    assert transporte.api_keys == ["cam-key-abc"]


def test_nao_envia_x_cypher_key_quando_api_key_omitida(tmp_path):
    transporte = TransporteFake()

    cliente(tmp_path, transporte).fetch("CAM-01", AMBIENTE)

    assert transporte.api_keys == [None]


def test_grava_cache_com_a_fonte_ainda_em_placeholder(tmp_path):
    cache = tmp_path / "cache.json"
    cliente(tmp_path, TransporteFake()).fetch("CAM-01", AMBIENTE)

    gravado = json.loads(cache.read_text())

    # O cache é um arquivo em disco: guardar a senha expandida nele desfaria
    # justamente a razão do placeholder existir.
    assert "${CAM01_USER}" in gravado["CAM-01"]["config"]["fonte"]
    assert "s3nh4" not in cache.read_text()


def test_api_fora_usa_o_cache(tmp_path):
    cliente(tmp_path, TransporteFake()).fetch("CAM-01", AMBIENTE)

    offline = cliente(tmp_path, TransporteFake(erro=ConnectionError("sem rede")))
    config = offline.fetch("CAM-01", AMBIENTE)

    assert config is not None
    assert config.origem == "cache"
    assert config.fonte == "rtsp://cypher:s3nh4@10.0.0.7/stream"


def test_cache_informa_a_propria_idade(tmp_path):
    cliente(tmp_path, TransporteFake()).fetch("CAM-01", AMBIENTE)

    depois = AGORA + timedelta(hours=3)
    offline = CameraConfigClient(
        "http://api/api/v1",
        cache_path=str(tmp_path / "cache.json"),
        transport=TransporteFake(erro=ConnectionError("sem rede")),
        now=lambda: depois,
    )

    assert offline.fetch("CAM-01", AMBIENTE).idade_segundos == pytest.approx(3 * 3600)


def test_api_fora_e_sem_cache_devolve_nada(tmp_path):
    offline = cliente(tmp_path, TransporteFake(erro=ConnectionError("sem rede")))

    assert offline.fetch("CAM-01", AMBIENTE) is None


def test_cache_de_outra_camera_nao_serve(tmp_path):
    cliente(tmp_path, TransporteFake()).fetch("CAM-01", AMBIENTE)

    offline = cliente(tmp_path, TransporteFake(erro=ConnectionError("sem rede")))

    assert offline.fetch("CAM-99", AMBIENTE) is None


def test_camera_desativada_aborta_e_nao_cai_para_o_cache(tmp_path):
    cliente(tmp_path, TransporteFake()).fetch("CAM-01", AMBIENTE)

    desativada = cliente(tmp_path, TransporteFake(
        status=409, corpo={"erro": "Camera desativada: CAM-01"}))

    # 409 é resposta definitiva do cadastro, não falha de rede: subir com o
    # cache reativaria na prática uma câmera tirada de operação.
    with pytest.raises(CameraDisabledError):
        desativada.fetch("CAM-01", AMBIENTE)


def test_camera_desativada_via_200_tambem_aborta(tmp_path):
    # O backend ainda não implementa o 409 dedicado nesta integração: um dia
    # pode simplesmente devolver 200 com "ativa": false no corpo. A garantia
    # de "nunca sobe câmera desativada" não pode depender de qual status HTTP
    # o servidor escolheu usar para dizer isso.
    ativa_false = cliente(tmp_path, TransporteFake(
        status=200, corpo={**CORPO, "ativa": False}))

    with pytest.raises(CameraDisabledError):
        ativa_false.fetch("CAM-01", AMBIENTE)


def test_camera_nao_cadastrada_aborta(tmp_path):
    nao_existe = cliente(tmp_path, TransporteFake(status=404, corpo={}))

    with pytest.raises(CameraNotRegisteredError):
        nao_existe.fetch("CAM-01", AMBIENTE)


def test_erro_inesperado_do_servidor_cai_para_o_cache(tmp_path):
    cliente(tmp_path, TransporteFake()).fetch("CAM-01", AMBIENTE)

    quebrado = cliente(tmp_path, TransporteFake(status=500, corpo={}))
    config = quebrado.fetch("CAM-01", AMBIENTE)

    # 500 é o servidor mal, não uma decisão sobre a câmera: degrada.
    assert config is not None
    assert config.origem == "cache"


def test_cache_corrompido_nao_derruba_o_pipeline(tmp_path):
    (tmp_path / "cache.json").write_text("{isso nao e json")

    offline = cliente(tmp_path, TransporteFake(erro=ConnectionError("sem rede")))

    assert offline.fetch("CAM-01", AMBIENTE) is None
