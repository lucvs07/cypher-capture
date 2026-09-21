from src.pose.epi_catalog import EpiTipo


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


class TestLabelDeViolacao:
    def test_cada_membro_tem_violation_label_com_prefixo_no(self):
        for tipo in EpiTipo:
            assert tipo.violation_label.startswith("no_")

    def test_labels_de_violacao_sao_unicas(self):
        labels = [tipo.violation_label for tipo in EpiTipo]
        assert len(labels) == len(set(labels))
