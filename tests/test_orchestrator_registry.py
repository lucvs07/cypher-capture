"""
Testes de ProcessRegistry (src/orchestrator/registry.py).

O que está sendo protegido: o registro precisa saber corretamente quais
câmeras têm processo vivo, com qual streamUrl cada uma foi iniciada (para o
Reconciler detectar mudança de config), e nunca deixar um processo "morto"
(poll() != None) passar por vivo. Lançador sempre injetado — nenhum teste
sobe um processo real do SO.
"""
from __future__ import annotations

import json

from src.orchestrator.registry import ProcessRegistry


class ProcessoFake:
    def __init__(self, pid: int):
        self.pid = pid
        self.codigo_saida = None
        self.terminado = False

    def poll(self):
        return self.codigo_saida

    def terminate(self):
        self.terminado = True

    def kill(self):
        self.terminado = True


class LauncherFake:
    def __init__(self):
        self.iniciados: list[tuple[str, str]] = []
        self._proximo_pid = 1000
        self.processos: dict[str, ProcessoFake] = {}

    def iniciar(self, camera_id: str, api_url: str) -> ProcessoFake:
        self.iniciados.append((camera_id, api_url))
        processo = ProcessoFake(pid=self._proximo_pid)
        self._proximo_pid += 1
        self.processos[camera_id] = processo
        return processo


def test_iniciar_registra_o_processo_e_marca_vivo(tmp_path):
    launcher = LauncherFake()
    registry = ProcessRegistry(launcher, state_path=str(tmp_path / "orchestrator-state.json"))

    registry.iniciar("CAM-01", "http://host:9001/stream", "http://api/api/v1")

    assert registry.esta_vivo("CAM-01") is True
    assert launcher.iniciados == [("CAM-01", "http://api/api/v1")]
    assert registry.stream_url_atual("CAM-01") == "http://host:9001/stream"


def test_esta_vivo_retorna_falso_para_camera_desconhecida(tmp_path):
    registry = ProcessRegistry(LauncherFake(), state_path=str(tmp_path / "orchestrator-state.json"))

    assert registry.esta_vivo("CAM-99") is False


def test_esta_vivo_retorna_falso_quando_o_processo_terminou(tmp_path):
    launcher = LauncherFake()
    registry = ProcessRegistry(launcher, state_path=str(tmp_path / "orchestrator-state.json"))
    registry.iniciar("CAM-01", "http://host:9001/stream", "http://api/api/v1")

    launcher.processos["CAM-01"].codigo_saida = 1

    assert registry.esta_vivo("CAM-01") is False


def test_encerrar_chama_terminate_e_remove_do_registro(tmp_path):
    launcher = LauncherFake()
    registry = ProcessRegistry(launcher, state_path=str(tmp_path / "orchestrator-state.json"))
    registry.iniciar("CAM-01", "http://host:9001/stream", "http://api/api/v1")

    registry.encerrar("CAM-01")

    assert launcher.processos["CAM-01"].terminado is True
    assert registry.esta_vivo("CAM-01") is False
    assert "CAM-01" not in registry.camera_ids_rastreados()


def test_encerrar_camera_desconhecida_nao_faz_nada(tmp_path):
    registry = ProcessRegistry(LauncherFake(), state_path=str(tmp_path / "orchestrator-state.json"))

    registry.encerrar("CAM-99")  # nao deve levantar excecao


def test_camera_ids_rastreados_reflete_o_que_esta_registrado(tmp_path):
    launcher = LauncherFake()
    registry = ProcessRegistry(launcher, state_path=str(tmp_path / "orchestrator-state.json"))
    registry.iniciar("CAM-01", "url-1", "http://api/api/v1")
    registry.iniciar("CAM-02", "url-2", "http://api/api/v1")

    assert sorted(registry.camera_ids_rastreados()) == ["CAM-01", "CAM-02"]


def test_iniciar_grava_o_pid_no_arquivo_de_estado(tmp_path):
    state_path = tmp_path / "orchestrator-state.json"
    launcher = LauncherFake()
    registry = ProcessRegistry(launcher, state_path=str(state_path))

    registry.iniciar("CAM-01", "url-1", "http://api/api/v1")

    estado = json.loads(state_path.read_text())
    assert estado == {"CAM-01": launcher.processos["CAM-01"].pid}


def test_encerrar_remove_a_camera_do_arquivo_de_estado(tmp_path):
    state_path = tmp_path / "orchestrator-state.json"
    launcher = LauncherFake()
    registry = ProcessRegistry(launcher, state_path=str(state_path))
    registry.iniciar("CAM-01", "url-1", "http://api/api/v1")

    registry.encerrar("CAM-01")

    estado = json.loads(state_path.read_text())
    assert estado == {}


def test_limpar_orfaos_mata_pid_vivo_registrado_no_boot_anterior(tmp_path):
    state_path = tmp_path / "orchestrator-state.json"
    state_path.write_text(json.dumps({"CAM-01": 999}))
    pids_mortos = []
    registry = ProcessRegistry(
        LauncherFake(), state_path=str(state_path),
        esta_pid_vivo=lambda pid: pid == 999,
        matar_pid=lambda pid: pids_mortos.append(pid),
    )

    registry.limpar_orfaos_do_boot_anterior()

    assert pids_mortos == [999]
    assert json.loads(state_path.read_text()) == {}


def test_limpar_orfaos_ignora_pid_ja_morto(tmp_path):
    state_path = tmp_path / "orchestrator-state.json"
    state_path.write_text(json.dumps({"CAM-01": 999}))
    pids_mortos = []
    registry = ProcessRegistry(
        LauncherFake(), state_path=str(state_path),
        esta_pid_vivo=lambda pid: False,
        matar_pid=lambda pid: pids_mortos.append(pid),
    )

    registry.limpar_orfaos_do_boot_anterior()

    assert pids_mortos == []


def test_limpar_orfaos_sem_arquivo_de_estado_nao_faz_nada(tmp_path):
    state_path = tmp_path / "orchestrator-state.json"
    registry = ProcessRegistry(LauncherFake(), state_path=str(state_path))

    registry.limpar_orfaos_do_boot_anterior()  # nao deve levantar excecao


def test_limpar_orfaos_com_arquivo_de_estado_corrompido_trata_como_vazio(tmp_path):
    """Um crash exatamente durante a escrita nao-atomica de _gravar_estado
    pode deixar o arquivo truncado/invalido -- isso precisa degradar
    graciosamente (como se nao houvesse estado anterior), nao derrubar o
    boot do orquestrador."""
    state_path = tmp_path / "orchestrator-state.json"
    state_path.write_text("{not valid json")
    registry = ProcessRegistry(LauncherFake(), state_path=str(state_path))

    registry.limpar_orfaos_do_boot_anterior()  # nao deve levantar excecao

    assert json.loads(state_path.read_text()) == {}
