"""
Testes de Reconciler (src/orchestrator/reconciler.py).

O que está sendo protegido: os seis passos do ciclo de reconciliação
(design doc, §2.3) — sobe câmera nova, derruba a que saiu do catálogo,
reinicia a que mudou de streamUrl, respeita backoff antes de reiniciar a
que morreu sozinha, respeita o limite de concorrência, e não toca em nada
quando o catálogo está inacessível. Usa o ProcessRegistry REAL (já testado
nas Tasks 2-3) com um LauncherFake, e uma sessão fake — assim o teste cobre
a decisão do Reconciler, confiando na mecânica do registro já provada.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.orchestrator.registry import ProcessRegistry
from src.orchestrator.reconciler import Reconciler
from src.orchestrator.session import CameraDesejada, CatalogoInacessivelError


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


class SessaoFake:
    def __init__(self, cameras=None, falhar=False):
        self.cameras = cameras if cameras is not None else []
        self.falhar = falhar

    def listar_cameras(self):
        if self.falhar:
            raise CatalogoInacessivelError("catalogo fora do ar")
        return self.cameras


class RelogioFake:
    def __init__(self):
        self.valor = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.valor

    def avancar(self, segundos: float) -> None:
        self.valor += timedelta(seconds=segundos)


def _reconciler(sessao, launcher=None, max_concorrentes=4, backoff_base=0.5,
                 backoff_max=10.0, relogio=None, state_path="outputs/x.json"):
    launcher = launcher or LauncherFake()
    registry = ProcessRegistry(launcher, state_path=state_path)
    relogio = relogio or RelogioFake()
    reconciler = Reconciler(sessao, registry, "http://api/api/v1", max_concorrentes,
                             backoff_base, backoff_max, agora=relogio)
    return reconciler, registry, launcher, relogio


def test_camera_nova_e_iniciada(tmp_path):
    sessao = SessaoFake(cameras=[CameraDesejada("CAM-01", "url-1")])
    reconciler, registry, launcher, _ = _reconciler(sessao, state_path=str(tmp_path / "s.json"))

    reconciler.reconciliar_uma_vez()

    assert launcher.iniciados == [("CAM-01", "http://api/api/v1")]
    assert registry.esta_vivo("CAM-01") is True


def test_camera_removida_do_catalogo_e_encerrada(tmp_path):
    sessao = SessaoFake(cameras=[CameraDesejada("CAM-01", "url-1")])
    reconciler, registry, launcher, _ = _reconciler(sessao, state_path=str(tmp_path / "s.json"))
    reconciler.reconciliar_uma_vez()

    sessao.cameras = []  # desativada/deletada
    reconciler.reconciliar_uma_vez()

    assert launcher.processos["CAM-01"].terminado is True
    assert registry.esta_vivo("CAM-01") is False


def test_streamurl_mudou_reinicia_com_a_nova_url(tmp_path):
    sessao = SessaoFake(cameras=[CameraDesejada("CAM-01", "url-antiga")])
    reconciler, registry, launcher, _ = _reconciler(sessao, state_path=str(tmp_path / "s.json"))
    reconciler.reconciliar_uma_vez()
    processo_antigo = launcher.processos["CAM-01"]  # captura antes do restart, o dict sera sobrescrito

    sessao.cameras = [CameraDesejada("CAM-01", "url-nova")]
    reconciler.reconciliar_uma_vez()

    assert processo_antigo.terminado is True  # o processo antigo foi encerrado
    assert launcher.iniciados == [("CAM-01", "http://api/api/v1"), ("CAM-01", "http://api/api/v1")]
    assert registry.stream_url_atual("CAM-01") == "url-nova"


def test_catalogo_inacessivel_nao_mexe_em_nada(tmp_path):
    sessao = SessaoFake(cameras=[CameraDesejada("CAM-01", "url-1")])
    reconciler, registry, launcher, _ = _reconciler(sessao, state_path=str(tmp_path / "s.json"))
    reconciler.reconciliar_uma_vez()

    sessao.falhar = True
    reconciler.reconciliar_uma_vez()

    assert len(launcher.iniciados) == 1  # nao tentou iniciar de novo
    assert registry.esta_vivo("CAM-01") is True  # continua rodando


def test_processo_morto_espera_o_backoff_antes_de_reiniciar(tmp_path):
    sessao = SessaoFake(cameras=[CameraDesejada("CAM-01", "url-1")])
    relogio = RelogioFake()
    reconciler, registry, launcher, relogio = _reconciler(
        sessao, backoff_base=1.0, backoff_max=10.0, relogio=relogio, state_path=str(tmp_path / "s.json"))
    reconciler.reconciliar_uma_vez()

    launcher.processos["CAM-01"].codigo_saida = 1  # morreu sozinha

    reconciler.reconciliar_uma_vez()
    assert len(launcher.iniciados) == 1  # ainda nao reiniciou -- backoff_base ainda nao passou

    relogio.avancar(1.0)
    reconciler.reconciliar_uma_vez()
    assert len(launcher.iniciados) == 2  # backoff passou -- reiniciou


def test_limite_de_concorrencia_deixa_o_excedente_de_fora(tmp_path):
    sessao = SessaoFake(cameras=[CameraDesejada("CAM-01", "url-1"), CameraDesejada("CAM-02", "url-2")])
    reconciler, registry, launcher, _ = _reconciler(
        sessao, max_concorrentes=1, state_path=str(tmp_path / "s.json"))

    reconciler.reconciliar_uma_vez()

    assert len(launcher.iniciados) == 1
    assert launcher.iniciados[0][0] == "CAM-01"  # ordenado por camera_id
    assert registry.esta_vivo("CAM-02") is False


def test_vaga_liberada_e_ocupada_no_ciclo_seguinte(tmp_path):
    sessao = SessaoFake(cameras=[CameraDesejada("CAM-01", "url-1"), CameraDesejada("CAM-02", "url-2")])
    reconciler, registry, launcher, _ = _reconciler(
        sessao, max_concorrentes=1, state_path=str(tmp_path / "s.json"))
    reconciler.reconciliar_uma_vez()  # so CAM-01 sobe

    sessao.cameras = [CameraDesejada("CAM-02", "url-2")]  # CAM-01 saiu do catalogo
    reconciler.reconciliar_uma_vez()

    assert registry.esta_vivo("CAM-01") is False
    assert registry.esta_vivo("CAM-02") is True


def test_log_resumo_do_ciclo_quando_camera_iniciada(tmp_path, caplog):
    sessao = SessaoFake(cameras=[CameraDesejada("CAM-01", "url-1")])
    reconciler, registry, launcher, _ = _reconciler(sessao, state_path=str(tmp_path / "s.json"))

    caplog.set_level(logging.INFO)
    reconciler.reconciliar_uma_vez()

    assert any(
        "Ciclo de reconciliacao" in r.getMessage()
        and "reiniciadas=" in r.getMessage()
        for r in caplog.records
        if r.name == "src.orchestrator.reconciler" and r.levelno == logging.INFO
    )


def test_nao_loga_resumo_quando_catalogo_inacessivel(tmp_path, caplog):
    sessao = SessaoFake(cameras=[CameraDesejada("CAM-01", "url-1")])
    reconciler, registry, launcher, _ = _reconciler(sessao, state_path=str(tmp_path / "s.json"))
    reconciler.reconciliar_uma_vez()

    sessao.falhar = True
    caplog.set_level(logging.INFO)
    caplog.clear()
    reconciler.reconciliar_uma_vez()

    assert not any(
        "Ciclo de reconciliacao" in r.getMessage()
        for r in caplog.records
        if r.name == "src.orchestrator.reconciler" and r.levelno == logging.INFO
    )
