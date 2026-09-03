"""
Testes de OrchestratorSession (src/orchestrator/session.py).

O que está sendo protegido: a sessão do orquestrador precisa sobreviver ao
tempo de vida inteiro do processo (diferente do ApiKeyProvisioner, que loga
uma vez e descarta) — então precisa re-logar sozinha quando a sessão expira,
e nunca deixar uma falha de sessão virar uma exceção não tratada que derruba
o orquestrador inteiro (isso é responsabilidade do Reconciler, mas a Session
precisa entregar um único tipo de erro previsível: CatalogoInacessivelError).
Transporte sempre injetado — nenhum teste toca rede.
"""
from __future__ import annotations

import pytest

from src.orchestrator.session import (
    CameraDesejada,
    CatalogoInacessivelError,
    CredenciaisAusentesError,
    LoginFalhouError,
    OrchestratorSession,
    SessaoExpiradaError,
)


class TransporteFake:
    """Registra logins e listagens; permite simular falha de login, sessão
    expirada na N-ésima chamada de listar_cameras, e catálogo inacessível."""

    def __init__(self, cameras=None, falhar_login=False,
                 expirar_na_chamada=None, falhar_apos_relogin=False):
        self.cameras = cameras if cameras is not None else []
        self.falhar_login = falhar_login
        self.expirar_na_chamada = expirar_na_chamada
        self.falhar_apos_relogin = falhar_apos_relogin
        self.logins: list[tuple[str, str, str]] = []
        self.chamadas_listar = 0

    def login(self, api_url: str, username: str, password: str) -> None:
        self.logins.append((api_url, username, password))
        if self.falhar_login:
            raise LoginFalhouError("login falhou")

    def listar_cameras(self, api_url: str) -> list[dict]:
        self.chamadas_listar += 1
        if self.chamadas_listar == self.expirar_na_chamada:
            raise SessaoExpiradaError()
        if self.falhar_apos_relogin and self.chamadas_listar == 2:
            raise SessaoExpiradaError()
        return self.cameras


def test_login_sem_credenciais_no_ambiente_levanta_erro():
    sessao = OrchestratorSession("http://api/api/v1", TransporteFake())

    with pytest.raises(CredenciaisAusentesError):
        sessao.login({})


def test_login_chama_transporte_com_credenciais_do_ambiente():
    transporte = TransporteFake()
    sessao = OrchestratorSession("http://api/api/v1", transporte)

    sessao.login({"CYPHER_GESTOR_USER": "orquestrador@acme", "CYPHER_GESTOR_PASSWORD": "senha123"})

    assert transporte.logins == [("http://api/api/v1", "orquestrador@acme", "senha123")]


def test_login_repassa_falha_do_transporte():
    transporte = TransporteFake(falhar_login=True)
    sessao = OrchestratorSession("http://api/api/v1", transporte)

    with pytest.raises(LoginFalhouError):
        sessao.login({"CYPHER_GESTOR_USER": "orquestrador@acme", "CYPHER_GESTOR_PASSWORD": "senha123"})


def test_listar_cameras_parseia_a_resposta():
    transporte = TransporteFake(cameras=[
        {"id": "CAM-01", "nome": "Entrada", "estacaoId": "EST-01", "streamUrl": "http://host:9001/stream"},
    ])
    sessao = OrchestratorSession("http://api/api/v1", transporte)
    sessao.login({"CYPHER_GESTOR_USER": "orquestrador@acme", "CYPHER_GESTOR_PASSWORD": "senha123"})

    cameras = sessao.listar_cameras()

    assert cameras == [CameraDesejada(camera_id="CAM-01", stream_url="http://host:9001/stream")]


def test_listar_cameras_reloga_uma_vez_quando_a_sessao_expira():
    transporte = TransporteFake(
        cameras=[{"id": "CAM-01", "nome": "Entrada", "estacaoId": "EST-01", "streamUrl": "http://host:9001/stream"}],
        expirar_na_chamada=1,
    )
    sessao = OrchestratorSession("http://api/api/v1", transporte)
    sessao.login({"CYPHER_GESTOR_USER": "orquestrador@acme", "CYPHER_GESTOR_PASSWORD": "senha123"})

    cameras = sessao.listar_cameras()

    assert cameras == [CameraDesejada(camera_id="CAM-01", stream_url="http://host:9001/stream")]
    assert len(transporte.logins) == 2  # login inicial + relogin apos a sessao expirar


def test_listar_cameras_propaga_catalogo_inacessivel_quando_o_relogin_tambem_falha():
    transporte = TransporteFake(
        cameras=[{"id": "CAM-01", "nome": "Entrada", "estacaoId": "EST-01", "streamUrl": "http://host:9001/stream"}],
        expirar_na_chamada=1,
        falhar_apos_relogin=True,
    )
    sessao = OrchestratorSession("http://api/api/v1", transporte)
    sessao.login({"CYPHER_GESTOR_USER": "orquestrador@acme", "CYPHER_GESTOR_PASSWORD": "senha123"})

    with pytest.raises(CatalogoInacessivelError):
        sessao.listar_cameras()
