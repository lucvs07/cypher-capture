"""
Testes do provisionamento automático da apiKey de ingestão.

O que está sendo protegido: cache-first (nunca regenera a chave se já
tiver uma gravada), credencial de bootstrap vem só do ambiente (nunca do
YAML), e erros de login/regenerate viram exceções nomeadas, não uma
KeyError obscura. Transporte sempre injetado — nenhum teste toca a rede.
"""
from __future__ import annotations

import json

import pytest

from src.capture.apikey_provisioning import (
    ApiKeyProvisioner,
    ApiKeyProvisioningError,
    MissingGestorCredentialError,
)


class TransporteFake:
    """Registra login/regenerate e permite simular falhas de cada etapa."""

    def __init__(self, chave: str = "cam-key-nova", falhar_login: bool = False,
                 falhar_regenerate: bool = False) -> None:
        self.chave = chave
        self.falhar_login = falhar_login
        self.falhar_regenerate = falhar_regenerate
        self.logins: list[tuple[str, str, str]] = []
        self.regenerates: list[tuple[str, str]] = []
        self._logado = False

    def login(self, api_url: str, username: str, password: str) -> None:
        self.logins.append((api_url, username, password))
        if self.falhar_login:
            raise ApiKeyProvisioningError("login falhou")
        self._logado = True

    def regenerate_api_key(self, api_url: str, camera_id: str) -> str:
        self.regenerates.append((api_url, camera_id))
        if self.falhar_regenerate:
            raise ApiKeyProvisioningError("regenerate falhou")
        assert self._logado, "regenerate chamado antes de login"
        return self.chave


def test_sem_cache_loga_e_regenera_gravando_o_resultado(tmp_path):
    cache = tmp_path / "apikey-cache.json"
    transporte = TransporteFake(chave="cam-key-abc123")
    provisioner = ApiKeyProvisioner("http://api/api/v1", cache_path=str(cache), transport=transporte)

    chave = provisioner.obter("CAM-01", {
        "CYPHER_GESTOR_USER": "gestor.demo", "CYPHER_GESTOR_PASSWORD": "senha123",
    })

    assert chave == "cam-key-abc123"
    assert transporte.logins == [("http://api/api/v1", "gestor.demo", "senha123")]
    assert transporte.regenerates == [("http://api/api/v1", "CAM-01")]
    assert json.loads(cache.read_text()) == {"CAM-01": "cam-key-abc123"}


def test_com_cache_nao_loga_de_novo(tmp_path):
    cache = tmp_path / "apikey-cache.json"
    cache.write_text(json.dumps({"CAM-01": "cam-key-ja-provisionada"}))
    transporte = TransporteFake()
    provisioner = ApiKeyProvisioner("http://api/api/v1", cache_path=str(cache), transport=transporte)

    chave = provisioner.obter("CAM-01", {
        "CYPHER_GESTOR_USER": "gestor.demo", "CYPHER_GESTOR_PASSWORD": "senha123",
    })

    assert chave == "cam-key-ja-provisionada"
    assert transporte.logins == []
    assert transporte.regenerates == []


def test_cache_e_por_camera_outra_camera_ainda_provisiona(tmp_path):
    cache = tmp_path / "apikey-cache.json"
    cache.write_text(json.dumps({"CAM-01": "cam-key-01"}))
    transporte = TransporteFake(chave="cam-key-02")
    provisioner = ApiKeyProvisioner("http://api/api/v1", cache_path=str(cache), transport=transporte)

    chave = provisioner.obter("CAM-02", {
        "CYPHER_GESTOR_USER": "gestor.demo", "CYPHER_GESTOR_PASSWORD": "senha123",
    })

    assert chave == "cam-key-02"
    assert json.loads(cache.read_text()) == {"CAM-01": "cam-key-01", "CAM-02": "cam-key-02"}


def test_sem_credencial_de_gestor_no_ambiente_falha_alto(tmp_path):
    cache = tmp_path / "apikey-cache.json"
    provisioner = ApiKeyProvisioner("http://api/api/v1", cache_path=str(cache), transport=TransporteFake())

    with pytest.raises(MissingGestorCredentialError):
        provisioner.obter("CAM-01", {})


def test_login_com_credencial_invalida_propaga_o_erro(tmp_path):
    cache = tmp_path / "apikey-cache.json"
    transporte = TransporteFake(falhar_login=True)
    provisioner = ApiKeyProvisioner("http://api/api/v1", cache_path=str(cache), transport=transporte)

    with pytest.raises(ApiKeyProvisioningError):
        provisioner.obter("CAM-01", {
            "CYPHER_GESTOR_USER": "gestor.demo", "CYPHER_GESTOR_PASSWORD": "errada",
        })
    # Falha antes do regenerate, e nada e gravado no cache.
    assert transporte.regenerates == []
    assert not cache.exists()


def test_camera_inexistente_no_regenerate_propaga_o_erro(tmp_path):
    cache = tmp_path / "apikey-cache.json"
    transporte = TransporteFake(falhar_regenerate=True)
    provisioner = ApiKeyProvisioner("http://api/api/v1", cache_path=str(cache), transport=transporte)

    with pytest.raises(ApiKeyProvisioningError):
        provisioner.obter("CAM-99", {
            "CYPHER_GESTOR_USER": "gestor.demo", "CYPHER_GESTOR_PASSWORD": "senha123",
        })
    assert not cache.exists()
