"""
Saída de incidentes.

O laço de frames tem 200 ms de orçamento e a medição atual é 45 ms p95: uma
chamada HTTP síncrona dentro dele consumiria toda a folga. Por isso enfileirar
é O(1) e todo o I/O acontece numa thread daemon.

"Vital" significa que perder um incidente importa, então o caminho de retry e
o spill em disco fazem parte do mínimo, não do refinamento.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Protocol

from .incident import IncidentReport

log = logging.getLogger(__name__)


class RejeitadoPeloBackend(Exception):
    """O backend recusou o payload de vez (4xx que nao se resolve sozinho).

    Retentar nao adianta: o mesmo payload vai ser recusado de novo nesta
    execucao, no spill e em todo boot seguinte. So os status de
    _STATUS_DEFINITIVOS contam; qualquer outro erro segue o caminho normal
    de retry + spill. 404 fica de fora de proposito: um PATCH de fechamento
    que chega antes da abertura (que ficou no spill) volta 404 e se resolve
    quando a abertura for drenada, e uma api_url errada da 404 em tudo e se
    resolve com correcao de configuracao, como 401/403.
    """

    def __init__(self, status: int, corpo: str = "") -> None:
        self.status = status
        self.corpo = corpo
        super().__init__(f"HTTP {status}: {corpo}")


# Lista explicita em vez de "todo 4xx menos alguns": um status novo ou
# inesperado cai no retry, que no pior caso so custa tentativas.
_STATUS_DEFINITIVOS = {400, 409, 410, 413, 422}


def _checar_resposta(resposta) -> None:
    if resposta.status_code in _STATUS_DEFINITIVOS:
        raise RejeitadoPeloBackend(resposta.status_code, resposta.text[:500])
    resposta.raise_for_status()


class Transport(Protocol):
    def post(self, url: str, payload: dict, headers: Optional[dict] = None) -> None: ...
    def patch(self, url: str, payload: dict, headers: Optional[dict] = None) -> None: ...


class RequestsTransport:
    """Transporte real. Importa requests preguiçosamente para manter os
    testes livres da dependência."""

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    def post(self, url: str, payload: dict, headers: Optional[dict] = None) -> None:
        import requests
        resposta = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        _checar_resposta(resposta)

    def patch(self, url: str, payload: dict, headers: Optional[dict] = None) -> None:
        import requests
        resposta = requests.patch(url, json=payload, headers=headers, timeout=self.timeout)
        _checar_resposta(resposta)


class Uploader(Protocol):
    def upload(self, frame, public_id: str) -> str: ...


class CloudinaryUploader:
    """Embrulha src/utils/upload.py. Importa preguiçosamente para que os
    testes rodem sem credencial e sem a dependência carregada."""

    def __init__(self, folder: str = "cypher-incidents") -> None:
        self.folder = folder

    def upload(self, frame, public_id: str) -> str:
        import os
        import tempfile

        import cloudinary.uploader
        import cv2

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, frame)
            resposta = cloudinary.uploader.upload(
                tmp.name, folder=self.folder, public_id=public_id)
        os.unlink(tmp.name)
        return resposta["secure_url"]


class ConsoleIncidentSink:
    """Comportamento padrão sem API configurada: imprime e segue."""

    def open(self, relatorio: IncidentReport) -> None:
        print(f"\n[INCIDENTE ABERTO] {relatorio.incident_key} "
              f"tipo={relatorio.type} sev={relatorio.severity} :: {relatorio.reason}")

    def close(self, relatorio: IncidentReport) -> None:
        duracao = relatorio.duration_seconds or 0.0
        print(f"\n[INCIDENTE FECHADO] {relatorio.incident_key} "
              f"dur={duracao:.1f}s pico={relatorio.peak_severity} "
              f"motivo={relatorio.closed_reason}")

    def start(self) -> None:
        pass

    def stop(self, timeout: float = 5.0) -> None:
        pass


@dataclass
class _Item:
    kind: str                  # "open" | "close"
    incident_key: str
    severity: int
    payload: dict
    attempts: int = 0
    # Só existe em memória: o spill em disco nunca serializa o frame.
    snapshot_frame: object = None


class MissingApiKeyError(Exception):
    """A variável de ambiente referenciada por ``api_key_env`` não existe (ou
    está vazia) no ambiente informado."""

    def __init__(self, env_var: str) -> None:
        self.env_var = env_var
        super().__init__(
            "Variavel de ambiente ausente para a chave de API do pipeline: "
            f"{env_var}. Defina-a no .env do host (ou exporte antes de subir "
            "o pipeline)."
        )


def resolve_api_key(env_var: Optional[str], env: Mapping[str, str]) -> Optional[str]:
    """Resolve o valor de ``env_var`` no ambiente informado.

    Mesmo contrato de ``expand_placeholders`` (`src/capture/config_client.py`):
    o segredo nunca vem do YAML, só do ambiente do host, e a ausência (ou
    valor vazio) falha alto nomeando a variável — em vez de mandar a chave
    como string vazia, ou pior, deixar de mandar o header sem avisar.
    """
    if not env_var:
        return None
    valor = env.get(env_var)
    if not valor:
        raise MissingApiKeyError(env_var)
    return valor


class HttpIncidentSink:
    """Fila limitada + thread de envio, com retry e spill em disco."""

    def __init__(self, api_url: str,
                 queue_size: int = 256,
                 spill_path: str = "outputs/incidents.jsonl",
                 rejected_path: Optional[str] = None,
                 max_attempts: int = 5,
                 backoff_base: float = 0.5,
                 transport: Optional[Transport] = None,
                 uploader: Optional[Uploader] = None,
                 api_key_env: Optional[str] = None,
                 api_key: Optional[str] = None,
                 env: Optional[Mapping[str, str]] = None) -> None:
        self.api_url = api_url.rstrip("/")
        self.queue_size = queue_size
        self.spill_path = Path(spill_path)
        # Rejeitados ficam ao lado do spill, mas nunca sao drenados de volta
        # pra fila: estao ali pra alguem inspecionar, nao pra reenviar.
        self.rejected_path = (Path(rejected_path) if rejected_path is not None
                              else self.spill_path.with_name(
                                  self.spill_path.stem + ".rejected.jsonl"))
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.transport = transport or RequestsTransport()
        self.uploader = uploader

        # Resolvida uma vez, no boot: a chave nao muda no meio de uma
        # execucao, e falhar aqui (em vez de dentro da thread de envio, no
        # primeiro POST) e o que impede o pipeline de subir "quase"
        # configurado e so descobrir o erro quando o primeiro incidente
        # tenta sair. `api_key` explicito (ex.: vindo do provisionamento
        # automatico via ApiKeyProvisioner) vence `api_key_env` -- os dois
        # nunca sao resolvidos ao mesmo tempo.
        chave = api_key if api_key is not None else resolve_api_key(
            api_key_env, env if env is not None else os.environ)
        self._headers: Optional[dict[str, str]] = (
            {"X-Cypher-Key": chave} if chave is not None else None)

        self._items: list[_Item] = []
        self._lock = threading.Lock()
        self._nao_vazia = threading.Condition(self._lock)
        self._parar = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._dropped = 0
        self._em_voo = 0        # itens fora da fila, dentro do transporte

    # ------------------------------------------------------------------
    # API pública — chamada da thread do frame, precisa ser barata
    # ------------------------------------------------------------------

    def open(self, relatorio: IncidentReport) -> None:
        self._enfileirar(_Item("open", relatorio.incident_key,
                               relatorio.severity, relatorio.open_payload(),
                               snapshot_frame=relatorio.snapshot_frame))

    def close(self, relatorio: IncidentReport) -> None:
        payload = relatorio.close_payload()
        if not payload:
            return
        self._enfileirar(_Item("close", relatorio.incident_key,
                               relatorio.severity, payload))

    def start(self) -> None:
        self._drenar_spill()
        self._worker = threading.Thread(target=self._loop, name="incident-sink",
                                        daemon=True)
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        # Drena antes de desistir. run_pipeline chama tracker.flush() e logo em
        # seguida stop(): os fechamentos finais entram na fila microssegundos
        # antes desta chamada, e sem esta espera todos iriam para o spill —
        # o painel ficaria com o motivo de fechamento errado ate a execucao
        # seguinte drenar o arquivo.
        prazo = time.monotonic() + timeout
        while time.monotonic() < prazo:
            with self._lock:
                if not self._items and self._em_voo == 0:
                    break
            time.sleep(0.02)

        self._parar.set()
        with self._nao_vazia:
            self._nao_vazia.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=timeout)
        # O que não foi entregue vai para o disco em vez de sumir.
        with self._lock:
            restantes = list(self._items)
            self._items.clear()
        for item in restantes:
            self._gravar_spill(item)

    def pending(self) -> int:
        with self._lock:
            return len(self._items)

    def dropped(self) -> int:
        return self._dropped

    def snapshot(self) -> list[_Item]:
        with self._lock:
            return list(self._items)

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------

    def _enfileirar(self, item: _Item) -> None:
        with self._nao_vazia:
            if len(self._items) >= self.queue_size:
                # Descarta o de MENOR severidade, nunca o mais recente: uma
                # queda não pode ser perdida por backlog de alerta ergonômico.
                indice = min(range(len(self._items)),
                             key=lambda i: self._items[i].severity)
                if self._items[indice].severity < item.severity:
                    descartado = self._items.pop(indice)
                    self._gravar_spill(descartado)
                else:
                    self._gravar_spill(item)
                    self._dropped += 1
                    log.warning("Fila cheia: incidente %s gravado em disco (descartes=%d)",
                                item.incident_key, self._dropped)
                    return
                self._dropped += 1
            self._items.append(item)
            self._nao_vazia.notify()

    def _loop(self) -> None:
        while not self._parar.is_set():
            with self._nao_vazia:
                while not self._items and not self._parar.is_set():
                    self._nao_vazia.wait(timeout=0.2)
                if self._parar.is_set() and not self._items:
                    return
                item = self._items.pop(0)
                self._em_voo += 1
            try:
                self._entregar(item)
            finally:
                with self._lock:
                    self._em_voo -= 1

    def _entregar(self, item: _Item) -> None:
        while item.attempts < self.max_attempts and not self._parar.is_set():
            item.attempts += 1
            try:
                if item.kind == "open":
                    self.transport.post(f"{self.api_url}/incidents", item.payload,
                                        headers=self._headers)
                    self._anexar_snapshot(item)
                else:
                    self.transport.patch(
                        f"{self.api_url}/incidents/{item.incident_key}", item.payload,
                        headers=self._headers)
                return
            except RejeitadoPeloBackend as rejeicao:
                log.error("Incidente %s (%s) recusado pelo backend, nao sera retentado "
                          "(gravado em %s): %s",
                          item.incident_key, item.kind, self.rejected_path, rejeicao)
                self._gravar_rejeitado(item, rejeicao)
                return
            except Exception as falha:
                if item.attempts >= self.max_attempts:
                    log.warning("Incidente %s nao entregue apos %d tentativas: %s",
                                item.incident_key, item.attempts, falha)
                    self._gravar_spill(item)
                    return
                time.sleep(self.backoff_base * (2 ** (item.attempts - 1)))

        # Saiu do laço sem entregar porque a parada chegou no meio: o item já
        # foi tirado da fila, então stop() não o alcança mais. Vai para o disco.
        self._gravar_spill(item)

    def _anexar_snapshot(self, item: _Item) -> None:
        """Upload e PATCH depois que o incidente já foi entregue.

        Falha aqui perde a imagem, nunca o alerta: o POST já aconteceu.
        """
        if self.uploader is None or item.snapshot_frame is None:
            return
        try:
            url = self.uploader.upload(item.snapshot_frame, item.incident_key)
            self.transport.patch(
                f"{self.api_url}/incidents/{item.incident_key}", {"snapshotUrl": url},
                headers=self._headers)
        except Exception as falha:
            log.warning("Snapshot de %s nao anexado: %s", item.incident_key, falha)

    def _gravar_spill(self, item: _Item) -> None:
        try:
            self.spill_path.parent.mkdir(parents=True, exist_ok=True)
            with self.spill_path.open("a") as arquivo:
                arquivo.write(json.dumps({
                    "kind": item.kind,
                    "incident_key": item.incident_key,
                    "severity": item.severity,
                    "payload": item.payload,
                }) + "\n")
        except OSError as erro:
            log.error("Falha ao gravar spill de %s: %s", item.incident_key, erro)

    def _gravar_rejeitado(self, item: _Item, rejeicao: RejeitadoPeloBackend) -> None:
        try:
            self.rejected_path.parent.mkdir(parents=True, exist_ok=True)
            with self.rejected_path.open("a") as arquivo:
                arquivo.write(json.dumps({
                    "kind": item.kind,
                    "incident_key": item.incident_key,
                    "status": rejeicao.status,
                    "resposta": rejeicao.corpo,
                    "payload": item.payload,
                }) + "\n")
        except OSError as erro:
            log.error("Falha ao gravar rejeitado de %s: %s", item.incident_key, erro)

    def _drenar_spill(self) -> None:
        """Reenfileira o que ficou pendente de uma execução anterior."""
        if not self.spill_path.exists():
            return
        linhas = [l for l in self.spill_path.read_text().splitlines() if l.strip()]
        self.spill_path.write_text("")
        for linha in linhas:
            try:
                registro = json.loads(linha)
            except json.JSONDecodeError:
                continue
            self._enfileirar(_Item(registro["kind"], registro.get("incident_key", ""),
                                   registro.get("severity", 0), registro["payload"]))
