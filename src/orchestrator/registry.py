"""
Registro de processos gerenciados pelo orquestrador (design doc, §2.2/§2.4).

Sabe o estado real (quais câmeras têm processo vivo, com qual streamUrl
cada uma foi iniciada) e persiste o mínimo necessário (PID por câmera) para
sobreviver a um restart do próprio orquestrador sem duplicar processo —
ver `limpar_orfaos_do_boot_anterior`.
"""
from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol


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


def _pid_esta_vivo_real(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # existe, mas de outro usuario -- trata como vivo
    return True


def _matar_pid_real(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class ProcessRegistry:
    def __init__(self, launcher: ProcessLauncher,
                 state_path: str = "outputs/orchestrator-state.json",
                 esta_pid_vivo: Callable[[int], bool] = _pid_esta_vivo_real,
                 matar_pid: Callable[[int], None] = _matar_pid_real) -> None:
        self.launcher = launcher
        self.state_path = Path(state_path)
        self.esta_pid_vivo = esta_pid_vivo
        self.matar_pid = matar_pid
        self._entradas: dict[str, _Entrada] = {}

    def iniciar(self, camera_id: str, stream_url: str, api_url: str) -> None:
        processo = self.launcher.iniciar(camera_id, api_url)
        self._entradas[camera_id] = _Entrada(stream_url=stream_url, processo=processo)
        self._gravar_estado()

    def encerrar(self, camera_id: str) -> None:
        entrada = self._entradas.pop(camera_id, None)
        if entrada is None:
            return
        entrada.processo.terminate()
        self._gravar_estado()

    def esta_vivo(self, camera_id: str) -> bool:
        entrada = self._entradas.get(camera_id)
        return entrada is not None and entrada.processo.poll() is None

    def stream_url_atual(self, camera_id: str) -> Optional[str]:
        entrada = self._entradas.get(camera_id)
        return entrada.stream_url if entrada else None

    def camera_ids_rastreados(self) -> list[str]:
        return list(self._entradas.keys())

    def limpar_orfaos_do_boot_anterior(self) -> None:
        if not self.state_path.exists():
            return
        estado_anterior: dict[str, int] = json.loads(self.state_path.read_text())
        for pid in estado_anterior.values():
            if self.esta_pid_vivo(pid):
                self.matar_pid(pid)
        self.state_path.write_text(json.dumps({}))

    def _gravar_estado(self) -> None:
        estado = {camera_id: entrada.processo.pid for camera_id, entrada in self._entradas.items()}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(estado))
