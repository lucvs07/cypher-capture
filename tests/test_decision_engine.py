"""
DecisionEngine: associacao EPI -> pessoa (unico ponto que ve o label cru do
detector) e fusao com a inferencia de ausencia por debounce
(epi_missing_confirmed) — achado de campo 2026-09-20.
"""
from src.fusion.engine import DecisionEngine
from src.pose.classifier import PostureCategory, PostureState
from src.pose.epi_catalog import EpiTipo
from src.zones.detector import ZoneResult, ZoneStatus
from src.zones.manager import Zone


def estado_normal() -> PostureState:
    return PostureState(category=PostureCategory.NORMAL, reason="")


class Reba:
    final_score = 1
    action_level = 1


def test_epi_missing_confirmed_entra_em_epi_violations():
    engine = DecisionEngine()

    event = engine.fuse(1, estado_normal(), Reba(), [], epi_present=[],
                         epi_missing_confirmed=[EpiTipo.HELMET])

    assert event.epi_violations == [EpiTipo.HELMET]


def test_epi_missing_confirmed_ausente_ou_none_nao_quebra():
    engine = DecisionEngine()

    event = engine.fuse(1, estado_normal(), Reba(), [], epi_present=[])

    assert event.epi_violations == []


def test_severidade_sobe_quando_ha_epi_ausente_mesmo_com_postura_normal():
    engine = DecisionEngine()

    event = engine.fuse(1, estado_normal(), Reba(), [], epi_present=[],
                         epi_missing_confirmed=[EpiTipo.HELMET])

    assert event.severity >= 2


def test_epi_missing_confirmed_duplicado_aparece_uma_vez():
    engine = DecisionEngine()

    event = engine.fuse(1, estado_normal(), Reba(), [], epi_present=[],
                         epi_missing_confirmed=[EpiTipo.HELMET, EpiTipo.HELMET])

    assert event.epi_violations == [EpiTipo.HELMET]


def zona_resultado(zone_type: str) -> ZoneResult:
    return ZoneResult(person_id=1, zone_id="z1", zone_name="Teste", status=ZoneStatus.INSIDE,
                       distance_px=0.0, severity=3, zone_type=zone_type)


class TestSeveridadePorTipoDeZona:
    """Achado de campo 2026-09-20: uma zona 'attention' (area de trabalho)
    nao e zona de risco — so 'restricted' deve elevar severidade so por a
    pessoa entrar. Antes disso o tipo da zona nao era checado em lugar
    nenhum, e ambas se comportavam identico."""

    def test_zona_restricted_eleva_severidade_a_3_so_por_entrar(self):
        engine = DecisionEngine()

        event = engine.fuse(1, estado_normal(), Reba(), [zona_resultado("restricted")], epi_present=[])

        assert event.severity == 3

    def test_zona_attention_nao_eleva_severidade_so_por_entrar(self):
        engine = DecisionEngine()

        event = engine.fuse(1, estado_normal(), Reba(), [zona_resultado("attention")], epi_present=[])

        assert event.severity == 0

    def test_zona_attention_ainda_eleva_severidade_se_epi_estiver_ausente(self):
        engine = DecisionEngine()

        event = engine.fuse(1, estado_normal(), Reba(), [zona_resultado("attention")],
                             epi_present=[], epi_missing_confirmed=[EpiTipo.HELMET])

        assert event.severity >= 2


def zona_com_epi(required_ppe: list[str]) -> ZoneResult:
    # Nomes como chegam do cadastro, resolvidos pela carga da Zone — o mesmo
    # caminho da zona vinda da API/zones.json ate o ZoneResult.
    # "attention" nao eleva severidade por presenca: o que sobra e so o 2 do
    # EPI ausente, e o +1 da zona fica visivel isolado (2 -> 3).
    zona = Zone.new("Teste", "attention", [(0, 0), (1, 0), (1, 1)], required_ppe=required_ppe)
    return ZoneResult(person_id=1, zone_id="z1", zone_name="Teste", status=ZoneStatus.INSIDE,
                       distance_px=0.0, severity=3, required_ppe=zona.required_ppe,
                       zone_type="attention")


class TestEscalonamentoPorEpiExigidoNaZona:
    """Taxonomia §4.2: estar na zona sem o EPI que ela exige soma +1. O front
    grava required_ppe com o nome canonico (HELMET, GLOVE, ...), mas o
    engine comparava f"no_{ppe}" — "no_HELMET" nunca batia com "no_helmet",
    e o escalonamento nunca disparava para zona cadastrada pela interface."""

    def test_nome_canonico_escala_severidade(self):
        engine = DecisionEngine()

        event = engine.fuse(1, estado_normal(), Reba(), [zona_com_epi(["HELMET"])],
                             epi_present=[], epi_missing_confirmed=[EpiTipo.HELMET])

        assert event.severity == 3

    def test_nome_canonico_com_label_de_detector_irregular_escala_severidade(self):
        # GLOVE/GOGGLES: nomes cujos labels antigos (no_gloves, no_goggle)
        # nao eram derivaveis do nome canonico.
        engine = DecisionEngine()

        event = engine.fuse(1, estado_normal(), Reba(), [zona_com_epi(["GLOVE", "GOGGLES"])],
                             epi_present=[], epi_missing_confirmed=[EpiTipo.GLOVE, EpiTipo.GOGGLES])

        assert event.severity == 3

    def test_nome_minusculo_do_zones_json_local_continua_escalando(self):
        engine = DecisionEngine()

        event = engine.fuse(1, estado_normal(), Reba(), [zona_com_epi(["helmet"])],
                             epi_present=[], epi_missing_confirmed=[EpiTipo.HELMET])

        assert event.severity == 3

    def test_epi_ausente_que_a_zona_nao_exige_nao_escala(self):
        engine = DecisionEngine()

        event = engine.fuse(1, estado_normal(), Reba(), [zona_com_epi(["VEST"])],
                             epi_present=[], epi_missing_confirmed=[EpiTipo.HELMET])

        assert event.severity == 2

    def test_nome_desconhecido_na_zona_e_ignorado(self):
        engine = DecisionEngine()

        event = engine.fuse(1, estado_normal(), Reba(), [zona_com_epi(["capacete"])],
                             epi_present=[], epi_missing_confirmed=[EpiTipo.HELMET])

        assert event.severity == 2


class TestAssociacaoEpiPessoa:
    PESSOA = {1: (0, 0, 100, 300)}

    def test_label_do_detector_vira_epi_tipo(self):
        assoc = DecisionEngine.associate_epis(
            self.PESSOA, [(10, 10, 40, 40), (10, 250, 40, 290)], ["HELMET", "boots"])

        assert assoc == {1: [EpiTipo.HELMET, EpiTipo.BOOTS]}

    def test_person_e_classe_desconhecida_sao_descartadas(self):
        assoc = DecisionEngine.associate_epis(
            self.PESSOA, [(0, 0, 100, 300), (10, 10, 40, 40)], ["Person", "no_helmet"])

        assert assoc == {1: []}

    def test_epi_fora_da_pessoa_nao_e_associado(self):
        assoc = DecisionEngine.associate_epis(self.PESSOA, [(200, 10, 240, 40)], ["HELMET"])

        assert assoc == {1: []}
