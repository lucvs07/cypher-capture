"""
Testes de ZoneConfigClient (src/zones/config_client.py).

O que está sendo protegido: a escada de degradação (API -> cache -> nada) e a
conversão normalizado->pixel usando a resolução real do frame — a mesma zona
normalizada precisa produzir polígonos corretos em resoluções diferentes
(design doc de zonas de risco, §3.1/§6). Nenhum teste toca rede: o
transporte é injetado, mesmo padrão de test_camera_config_client.py.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.zones.config_client import ZoneConfigClient

AGORA = datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc)

CORPO_API = [
    {
        "id": 1,
        "nome": "Area da prensa",
        "zoneType": "RESTRICTED",
        "polygon": [{"x": 0.25, "y": 0.25}, {"x": 0.75, "y": 0.25}, {"x": 0.75, "y": 0.75}],
        "severity": 4,
        "requiredPpe": ["capacete"],
    },
]


class TransporteFake:
    def __init__(self, status=200, corpo=None, erro=None):
        self.status = status
        self.corpo = corpo if corpo is not None else CORPO_API
        self.erro = erro
        self.urls: list[str] = []

    def get_json(self, url):
        self.urls.append(url)
        if self.erro is not None:
            raise self.erro
        return self.status, self.corpo


def cliente(tmp_path, transporte, **kwargs):
    return ZoneConfigClient(
        "http://api/api/v1",
        cache_path=str(tmp_path / "zones-cache.json"),
        transport=transporte,
        now=lambda: AGORA,
        **kwargs,
    )


def test_busca_no_endpoint_de_config_de_zonas_da_camera(tmp_path):
    transporte = TransporteFake()

    cliente(tmp_path, transporte).fetch("CAM-01", 1280, 720)

    assert transporte.urls == ["http://api/api/v1/cameras/CAM-01/zones/config"]


def test_converte_normalizado_para_pixel_usando_a_resolucao_informada(tmp_path):
    zonas = cliente(tmp_path, TransporteFake()).fetch("CAM-01", 1280, 720)

    assert len(zonas) == 1
    z = zonas[0]
    assert z.zone_id == "1"
    assert z.name == "Area da prensa"
    assert z.zone_type == "restricted"
    assert z.polygon == [(320.0, 180.0), (960.0, 180.0), (960.0, 540.0)]
    assert z.severity == 4
    assert z.required_ppe == ["capacete"]


def test_mesma_zona_normalizada_produz_poligono_diferente_em_outra_resolucao(tmp_path):
    zonas_720p = cliente(tmp_path, TransporteFake()).fetch("CAM-01", 1280, 720)
    zonas_1080p = cliente(tmp_path, TransporteFake()).fetch("CAM-01", 1920, 1080)

    assert zonas_720p[0].polygon == [(320.0, 180.0), (960.0, 180.0), (960.0, 540.0)]
    assert zonas_1080p[0].polygon == [(480.0, 270.0), (1440.0, 270.0), (1440.0, 810.0)]


def test_grava_cache_normalizado_nao_convertido(tmp_path):
    transporte = TransporteFake()
    cache_path = tmp_path / "zones-cache.json"

    ZoneConfigClient("http://api/api/v1", cache_path=str(cache_path),
                     transport=transporte, now=lambda: AGORA).fetch("CAM-01", 1280, 720)

    gravado = json.loads(cache_path.read_text())
    assert gravado["CAM-01"]["zones"] == CORPO_API


def test_api_fora_usa_cache_e_converte_com_a_resolucao_atual(tmp_path):
    cache_path = tmp_path / "zones-cache.json"
    cache_path.write_text(json.dumps({
        "CAM-01": {"fetched_at": AGORA.isoformat(), "zones": CORPO_API},
    }))

    zonas = ZoneConfigClient("http://api/api/v1", cache_path=str(cache_path),
                            transport=TransporteFake(erro=ConnectionError("fora do ar")),
                            now=lambda: AGORA + timedelta(hours=2)).fetch("CAM-01", 640, 360)

    assert zonas[0].polygon == [(160.0, 90.0), (480.0, 90.0), (480.0, 270.0)]


def test_api_fora_sem_cache_devolve_none(tmp_path):
    zonas = cliente(tmp_path, TransporteFake(erro=ConnectionError("fora do ar"))).fetch("CAM-01", 1280, 720)

    assert zonas is None


def test_zero_zonas_e_aceito(tmp_path):
    zonas = cliente(tmp_path, TransporteFake(corpo=[])).fetch("CAM-01", 1280, 720)

    assert zonas == []


def test_status_inesperado_da_api_sem_cache_devolve_none(tmp_path):
    zonas = cliente(tmp_path, TransporteFake(status=500, corpo={})).fetch("CAM-01", 1280, 720)

    assert zonas is None
