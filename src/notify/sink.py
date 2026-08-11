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
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from .incident import IncidentReport

log = logging.getLogger(__name__)


class Transport(Protocol):
    def post(self, url: str, payload: dict) -> None: ...
    def patch(self, url: str, payload: dict) -> None: ...


class RequestsTransport:
    """Transporte real. Importa requests preguiçosamente para manter os
    testes livres da dependência."""

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    def post(self, url: str, payload: dict) -> None:
        import requests
        resposta = requests.post(url, json=payload, timeout=self.timeout)
        resposta.raise_for_status()

    def patch(self, url: str, payload: dict) -> None:
        import requests
        resposta = requests.patch(url, json=payload, timeout=self.timeout)
        resposta.raise_for_status()


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


class HttpIncidentSink:
    """Fila limitada + thread de envio, com retry e spill em disco."""

    def __init__(self, api_url: str,
                 queue_size: int = 256,
                 spill_path: str = "outputs/incidents.jsonl",
                 max_attempts: int = 5,
                 backoff_base: float = 0.5,
                 transport: Optional[Transport] = None,
                 uploader: Optional[Uploader] = None) -> None:
        self.api_url = api_url.rstrip("/")
        self.queue_size = queue_size
        self.spill_path = Path(spill_path)
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.transport = transport or RequestsTransport()
        self.uploader = uploader

        self._items: list[_Item] = []
        self._lock = threading.Lock()
        self._nao_vazia = threading.Condition(self._lock)
        self._parar = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._dropped = 0

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
            self._entregar(item)

    def _entregar(self, item: _Item) -> None:
        while item.attempts < self.max_attempts and not self._parar.is_set():
            item.attempts += 1
            try:
                if item.kind == "open":
                    self.transport.post(f"{self.api_url}/incidents", item.payload)
                    self._anexar_snapshot(item)
                else:
                    self.transport.patch(
                        f"{self.api_url}/incidents/{item.incident_key}", item.payload)
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
                f"{self.api_url}/incidents/{item.incident_key}", {"snapshotUrl": url})
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
