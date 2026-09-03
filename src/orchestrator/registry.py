"""
Registro de processos gerenciados pelo orquestrador (design doc, §2.2).

Sabe o estado real (quais câmeras têm processo vivo, com qual streamUrl
cada uma foi iniciada) — a decisão de QUANDO iniciar/encerrar é do
Reconciler, não daqui.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


class ProcessoLike(Protocol):
    pid: int

    def poll(self) -> Optional[int]: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


class ProcessLauncher(Protocol):
    def iniciar(self, camera_id: str, api_url: str) -> ProcessoLike: ...


@dataclass
class _Entrada:
    stream_url: str
    processo: ProcessoLike


class ProcessRegistry:
    def __init__(self, launcher: ProcessLauncher) -> None:
        self.launcher = launcher
        self._entradas: dict[str, _Entrada] = {}

    def iniciar(self, camera_id: str, stream_url: str, api_url: str) -> None:
        processo = self.launcher.iniciar(camera_id, api_url)
        self._entradas[camera_id] = _Entrada(stream_url=stream_url, processo=processo)

    def encerrar(self, camera_id: str) -> None:
        entrada = self._entradas.pop(camera_id, None)
        if entrada is None:
            return
        entrada.processo.terminate()

    def esta_vivo(self, camera_id: str) -> bool:
        entrada = self._entradas.get(camera_id)
        return entrada is not None and entrada.processo.poll() is None

    def stream_url_atual(self, camera_id: str) -> Optional[str]:
        entrada = self._entradas.get(camera_id)
        return entrada.stream_url if entrada else None

    def camera_ids_rastreados(self) -> list[str]:
        return list(self._entradas.keys())
