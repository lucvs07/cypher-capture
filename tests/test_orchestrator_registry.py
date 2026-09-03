"""
Testes de ProcessRegistry (src/orchestrator/registry.py).

O que está sendo protegido: o registro precisa saber corretamente quais
câmeras têm processo vivo, com qual streamUrl cada uma foi iniciada (para o
Reconciler detectar mudança de config), e nunca deixar um processo "morto"
(poll() != None) passar por vivo. Lançador sempre injetado — nenhum teste
sobe um processo real do SO.
"""
from __future__ import annotations

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


def test_iniciar_registra_o_processo_e_marca_vivo():
    launcher = LauncherFake()
    registry = ProcessRegistry(launcher)

    registry.iniciar("CAM-01", "http://host:9001/stream", "http://api/api/v1")

    assert registry.esta_vivo("CAM-01") is True
    assert launcher.iniciados == [("CAM-01", "http://api/api/v1")]
    assert registry.stream_url_atual("CAM-01") == "http://host:9001/stream"


def test_esta_vivo_retorna_falso_para_camera_desconhecida():
    registry = ProcessRegistry(LauncherFake())

    assert registry.esta_vivo("CAM-99") is False


def test_esta_vivo_retorna_falso_quando_o_processo_terminou():
    launcher = LauncherFake()
    registry = ProcessRegistry(launcher)
    registry.iniciar("CAM-01", "http://host:9001/stream", "http://api/api/v1")

    launcher.processos["CAM-01"].codigo_saida = 1

    assert registry.esta_vivo("CAM-01") is False


def test_encerrar_chama_terminate_e_remove_do_registro():
    launcher = LauncherFake()
    registry = ProcessRegistry(launcher)
    registry.iniciar("CAM-01", "http://host:9001/stream", "http://api/api/v1")

    registry.encerrar("CAM-01")

    assert launcher.processos["CAM-01"].terminado is True
    assert registry.esta_vivo("CAM-01") is False
    assert "CAM-01" not in registry.camera_ids_rastreados()


def test_encerrar_camera_desconhecida_nao_faz_nada():
    registry = ProcessRegistry(LauncherFake())

    registry.encerrar("CAM-99")  # nao deve levantar excecao


def test_camera_ids_rastreados_reflete_o_que_esta_registrado():
    launcher = LauncherFake()
    registry = ProcessRegistry(launcher)
    registry.iniciar("CAM-01", "url-1", "http://api/api/v1")
    registry.iniciar("CAM-02", "url-2", "http://api/api/v1")

    assert sorted(registry.camera_ids_rastreados()) == ["CAM-01", "CAM-02"]
