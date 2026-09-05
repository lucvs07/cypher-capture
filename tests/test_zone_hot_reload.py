"""
Testes de ZoneHotReloader (src/zones/hot_reload.py).

O que está sendo protegido: uma zona criada ou editada pela UI só valia a
partir do próximo boot do pipeline (design doc de zonas de risco §3.4/§7 —
hot-reload registrado como fora de escopo original). Este reloader refaz a
busca periodicamente em segundo plano e só troca as zonas do pipeline quando
a lista realmente muda, sem travar o loop de frames com uma chamada de rede
síncrona. `tick()` é a unidade testável sem depender de threading real —
`start()`/`stop()` são cobertos por um teste de integração leve, com
intervalo real bem curto.
"""
from __future__ import annotations

import time

from src.zones.hot_reload import ZoneHotReloader
from src.zones.manager import Zone


class ClienteZonasFake:
    def __init__(self, respostas):
        # cada chamada de fetch() consome a próxima resposta da lista; a
        # última é repetida se o teste chamar tick() mais vezes do que
        # preparou respostas.
        self.respostas = list(respostas)
        self.chamadas: list[tuple[str, int, int, object]] = []

    def fetch(self, camera_id, width, height, api_key=None):
        self.chamadas.append((camera_id, width, height, api_key))
        if len(self.respostas) > 1:
            return self.respostas.pop(0)
        return self.respostas[0]


ZONA_A = Zone.new("A", "restricted", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
ZONA_B = Zone.new("B", "attention", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])


def test_tick_chama_on_zonas_quando_a_lista_muda():
    cliente = ClienteZonasFake([[ZONA_A]])
    aplicadas = []

    reloader = ZoneHotReloader(cliente, "CAM-01", 1280, 720, on_zonas=aplicadas.append)
    mudou = reloader.tick()

    assert mudou is True
    assert aplicadas == [[ZONA_A]]


def test_tick_nao_chama_on_zonas_quando_a_lista_e_igual():
    cliente = ClienteZonasFake([[ZONA_A], [ZONA_A]])
    aplicadas = []
    reloader = ZoneHotReloader(cliente, "CAM-01", 1280, 720, on_zonas=aplicadas.append)

    reloader.tick()
    mudou = reloader.tick()

    assert mudou is False
    assert aplicadas == [[ZONA_A]]


def test_tick_chama_on_zonas_de_novo_quando_a_lista_muda_depois():
    cliente = ClienteZonasFake([[ZONA_A], [ZONA_B]])
    aplicadas = []
    reloader = ZoneHotReloader(cliente, "CAM-01", 1280, 720, on_zonas=aplicadas.append)

    reloader.tick()
    mudou = reloader.tick()

    assert mudou is True
    assert aplicadas == [[ZONA_A], [ZONA_B]]


def test_tick_ignora_quando_o_cliente_nao_tem_nada():
    cliente = ClienteZonasFake([None])
    aplicadas = []
    reloader = ZoneHotReloader(cliente, "CAM-01", 1280, 720, on_zonas=aplicadas.append)

    mudou = reloader.tick()

    assert mudou is False
    assert aplicadas == []


def test_tick_repassa_api_key_e_resolucao_para_o_cliente():
    cliente = ClienteZonasFake([[ZONA_A]])
    reloader = ZoneHotReloader(cliente, "CAM-01", 1280, 720, on_zonas=lambda z: None,
                               api_key="cam-key-abc")

    reloader.tick()

    assert cliente.chamadas == [("CAM-01", 1280, 720, "cam-key-abc")]


def test_estado_inicial_pode_ser_prefixado_para_nao_reaplicar_o_que_o_boot_ja_aplicou():
    cliente = ClienteZonasFake([[ZONA_A]])
    aplicadas = []
    reloader = ZoneHotReloader(cliente, "CAM-01", 1280, 720, on_zonas=aplicadas.append)
    reloader.marcar_zonas_atuais([ZONA_A])

    mudou = reloader.tick()

    assert mudou is False
    assert aplicadas == []


def test_start_e_stop_rodam_tick_em_segundo_plano_e_encerram_de_forma_limpa():
    cliente = ClienteZonasFake([[ZONA_A]])
    aplicadas = []
    reloader = ZoneHotReloader(cliente, "CAM-01", 1280, 720, on_zonas=aplicadas.append,
                               interval_seconds=0.02)

    reloader.start()
    time.sleep(0.1)
    reloader.stop()

    assert aplicadas == [[ZONA_A]]
    assert len(cliente.chamadas) >= 1


def test_stop_antes_de_start_nao_levanta():
    cliente = ClienteZonasFake([[ZONA_A]])
    reloader = ZoneHotReloader(cliente, "CAM-01", 1280, 720, on_zonas=lambda z: None)

    reloader.stop()  # nao deve levantar


def test_erro_do_cliente_durante_o_loop_nao_derruba_a_thread():
    class ClienteQueFalhaUmaVez:
        def __init__(self):
            self.chamadas = 0

        def fetch(self, camera_id, width, height, api_key=None):
            self.chamadas += 1
            if self.chamadas == 1:
                raise RuntimeError("falha de rede simulada")
            return [ZONA_A]

    cliente = ClienteQueFalhaUmaVez()
    aplicadas = []
    reloader = ZoneHotReloader(cliente, "CAM-01", 1280, 720, on_zonas=aplicadas.append,
                               interval_seconds=0.02)

    reloader.start()
    time.sleep(0.15)
    reloader.stop()

    assert aplicadas == [[ZONA_A]]
    assert cliente.chamadas >= 2
