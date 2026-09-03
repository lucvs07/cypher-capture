"""
Sessão do orquestrador (design doc de orquestrador de pipelines, §2.1/§3).

Ao contrário de `ApiKeyProvisioner` (login uma vez, descarta), esta sessão
precisa sobreviver ao tempo de vida inteiro do orquestrador: faz polling
contínuo em `GET /cameras`, então precisa re-logar sozinha quando a sessão
expira. `listar_cameras()` absorve TODA falha de sessão/rede num único tipo
de exceção (`CatalogoInacessivelError`) — quem chama (o `Reconciler`) só
precisa tratar um caso: "não deu pra confirmar o catálogo agora".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol


class CredenciaisAusentesError(Exception):
    """CYPHER_GESTOR_USER/CYPHER_GESTOR_PASSWORD ausentes no ambiente do host."""

    def __init__(self) -> None:
        super().__init__(
            "CYPHER_GESTOR_USER e CYPHER_GESTOR_PASSWORD precisam estar "
            "definidos no ambiente (.env do host) para o orquestrador logar "
            "como a conta de servico GESTOR daquele tenant."
        )


class LoginFalhouError(Exception):
    """Login rejeitado pelo backend — credencial invalida."""


class SessaoExpiradaError(Exception):
    """GET /cameras devolveu 401 — a sessao do orquestrador expirou."""


class CatalogoInacessivelError(Exception):
    """Catalogo nao pode ser confirmado agora (rede, backend fora do ar, ou
    sessao expirou e o re-login tambem falhou). O Reconciler trata isso como
    'nao mexe em nada neste ciclo'."""


@dataclass(frozen=True)
class CameraDesejada:
    camera_id: str
    stream_url: str


class SessionTransport(Protocol):
    def login(self, api_url: str, username: str, password: str) -> None: ...
    def listar_cameras(self, api_url: str) -> list[dict]: ...


class RequestsSessionTransport:
    """Transporte real. Uma unica `requests.Session` mantem o cookie de
    sessao entre chamadas — mesmo padrao de `RequestsProvisioningTransport`
    em `apikey_provisioning.py`."""

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout
        self._session = None

    def login(self, api_url: str, username: str, password: str) -> None:
        import requests
        self._session = requests.Session()
        resposta = self._session.post(
            f"{api_url}/auth/login", json={"username": username, "password": password},
            timeout=self.timeout)
        if resposta.status_code != 200:
            raise LoginFalhouError(
                f"Login do orquestrador falhou ({resposta.status_code}) — "
                "confira CYPHER_GESTOR_USER/CYPHER_GESTOR_PASSWORD.")

    def listar_cameras(self, api_url: str) -> list[dict]:
        if self._session is None:
            raise CatalogoInacessivelError("listar_cameras chamado antes de login().")
        try:
            resposta = self._session.get(f"{api_url}/cameras", timeout=self.timeout)
        except Exception as erro:  # rede fora do ar, timeout, etc.
            raise CatalogoInacessivelError(f"GET /cameras falhou: {erro}") from erro
        if resposta.status_code == 401:
            raise SessaoExpiradaError()
        if resposta.status_code != 200:
            raise CatalogoInacessivelError(f"GET /cameras devolveu {resposta.status_code}")
        return resposta.json()


class OrchestratorSession:
    def __init__(self, api_url: str, transport: SessionTransport) -> None:
        self.api_url = api_url.rstrip("/")
        self.transport = transport
        self._username: Optional[str] = None
        self._password: Optional[str] = None

    def login(self, env: Mapping[str, str]) -> None:
        username = env.get("CYPHER_GESTOR_USER")
        password = env.get("CYPHER_GESTOR_PASSWORD")
        if not username or not password:
            raise CredenciaisAusentesError()
        self.transport.login(self.api_url, username, password)
        self._username, self._password = username, password

    def listar_cameras(self) -> list[CameraDesejada]:
        try:
            dados = self.transport.listar_cameras(self.api_url)
        except SessaoExpiradaError:
            try:
                self.transport.login(self.api_url, self._username, self._password)
                dados = self.transport.listar_cameras(self.api_url)
            except (SessaoExpiradaError, LoginFalhouError, CatalogoInacessivelError) as segunda_falha:
                raise CatalogoInacessivelError(
                    f"sessao expirou e o re-login falhou: {segunda_falha}") from segunda_falha
        return [CameraDesejada(camera_id=d["id"], stream_url=d["streamUrl"]) for d in dados]
