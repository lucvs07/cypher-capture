"""
Testes de ZoneManager (src/zones/manager.py) — cobrindo o construtor com
zonas pré-carregadas em memória, adicionado para a busca remota (design doc
de zonas de risco, §3.1). O caminho por arquivo já existia sem testes
dedicados; não é reescrito aqui, só preservado.
"""
from __future__ import annotations

from src.zones.manager import Zone, ZoneManager


def test_zones_pre_carregadas_nao_tocam_arquivo(tmp_path):
    caminho_inexistente = tmp_path / "nao-existe" / "zones.json"
    zona = Zone.new("Area da prensa", "restricted", [(100.0, 100.0), (400.0, 100.0), (400.0, 400.0)])

    manager = ZoneManager(str(caminho_inexistente), zones=[zona])

    assert manager.get_all() == [zona]
    assert not caminho_inexistente.exists()


def test_sem_zones_cai_no_carregamento_por_arquivo_como_antes(tmp_path):
    caminho = tmp_path / "zones.json"
    manager_para_gravar = ZoneManager(str(caminho))
    manager_para_gravar.add(Zone.new("Zona X", "attention", [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]))

    manager_lido = ZoneManager(str(caminho))

    assert len(manager_lido.get_all()) == 1
    assert manager_lido.get_all()[0].name == "Zona X"


def test_zones_vazia_explicita_nao_cai_no_arquivo_com_zonas(tmp_path):
    # `zones=[]` (lista vazia, nao None) precisa ser respeitado como "zero
    # zonas" mesmo com um arquivo populado no mesmo caminho — "if zones is
    # not None" e um keystroke de distancia de "if zones", que trataria
    # [] como falsy e cairia no arquivo por engano.
    caminho = tmp_path / "zones.json"
    manager_para_gravar = ZoneManager(str(caminho))
    manager_para_gravar.add(Zone.new("Zona X", "attention", [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]))

    manager = ZoneManager(str(caminho), zones=[])

    assert manager.get_all() == []


def test_zones_pre_carregadas_indexam_por_zone_id_como_o_carregamento_por_arquivo(tmp_path):
    z1 = Zone.new("A", "restricted", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    z2 = Zone.new("B", "attention", [(2.0, 2.0), (3.0, 2.0), (3.0, 3.0)])

    manager = ZoneManager(str(tmp_path / "zones.json"), zones=[z1, z2])

    assert {z.zone_id for z in manager.get_all()} == {z1.zone_id, z2.zone_id}
