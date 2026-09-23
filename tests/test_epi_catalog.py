import logging
from pathlib import Path

import pytest

from src.pose.epi_catalog import EpiTipo, resolve_wire_names

ROOT = Path(__file__).resolve().parent.parent


class TestResolucaoPorClasseDoDetector:
    def test_cada_membro_resolve_da_propria_detector_label(self):
        for tipo in EpiTipo:
            assert EpiTipo.from_detector_label(tipo.detector_label) is tipo

    def test_label_desconhecida_devolve_none(self):
        assert EpiTipo.from_detector_label("Person") is None
        assert EpiTipo.from_detector_label("algo_que_nao_existe") is None


class TestResolucaoPorNomeDoCadastro:
    def test_nome_exato_resolve(self):
        assert EpiTipo.from_wire_name("HELMET") is EpiTipo.HELMET

    def test_case_insensitive(self):
        assert EpiTipo.from_wire_name("helmet") is EpiTipo.HELMET
        assert EpiTipo.from_wire_name("Boots") is EpiTipo.BOOTS

    def test_espacos_nas_pontas_sao_ignorados(self):
        assert EpiTipo.from_wire_name("  vest  ") is EpiTipo.VEST

    def test_nome_desconhecido_devolve_none_em_vez_de_lancar(self):
        # Achado 2026-09-16: cadastro guardava nome livre em portugues
        # ("capacete") sem validacao nenhuma — resolver com None (nao
        # exception) deixa o chamador decidir ignorar o item, nao derrubar
        # o pipeline por um dado legado.
        assert EpiTipo.from_wire_name("capacete") is None

    @pytest.mark.parametrize("valor", [None, 3, ["HELMET"], {"x": 1}])
    def test_valor_que_nao_e_string_devolve_none(self, valor):
        assert EpiTipo.from_wire_name(valor) is None


class TestResolveWireNames:
    def test_resolve_deduplica_e_preserva_ordem(self):
        assert resolve_wire_names(["GLOVE", "helmet", "HELMET"], "zona Z") == [
            EpiTipo.GLOVE, EpiTipo.HELMET]

    def test_none_ou_vazio_vira_lista_vazia(self):
        assert resolve_wire_names(None, "zona Z") == []
        assert resolve_wire_names([], "zona Z") == []

    def test_nao_detectavel_e_descartado_com_aviso(self, caplog):
        with caplog.at_level(logging.WARNING):
            tipos = resolve_wire_names(["auricular", "VEST", None], "zona Z")
        assert tipos == [EpiTipo.VEST]
        assert "zona Z" in caplog.text
        assert "'auricular'" in caplog.text


PESOS = sorted((ROOT / "models").glob("*.pt"))


@pytest.mark.parametrize("pesos", PESOS, ids=lambda p: p.name)
def test_catalogo_bate_com_as_classes_do_modelo(pesos):
    """Retreino que renomeie uma classe (ex.: Goggles -> goggles) faria
    from_detector_label devolver None e o item seria sempre "ausente" —
    falso positivo de EPI em massa sem erro nenhum. Este teste pega isso."""
    ultralytics = pytest.importorskip("ultralytics")
    nomes = set(ultralytics.YOLO(str(pesos)).names.values()) - {"Person"}
    assert nomes == {tipo.detector_label for tipo in EpiTipo}
