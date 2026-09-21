"""
EpiAbsenceTracker: declara EPI ausente so depois de N frames seguidos sem
deteccao (achado de campo 2026-09-20 — bbox as vezes nao aparece por oclusao
ou blur; 1 frame perdido nao pode virar alerta de ausencia).
"""
from src.pose.epi_absence import EpiAbsenceTracker
from src.pose.epi_catalog import EpiTipo


def tracker(confirm_frames: int = 4, min_track_frames: int = 0) -> EpiAbsenceTracker:
    return EpiAbsenceTracker(confirm_frames=confirm_frames, min_track_frames=min_track_frames)


class TestConfirmacaoPorFramesConsecutivos:
    def test_ausencia_isolada_de_1_frame_nao_confirma(self):
        t = tracker(confirm_frames=4)
        ausentes = t.update(person_id=1, epi_present_labels=[], required=[EpiTipo.HELMET],
                             bbox_truncated=False)
        assert ausentes == []

    def test_ausencia_por_frames_consecutivos_suficientes_confirma(self):
        t = tracker(confirm_frames=3)
        for _ in range(2):
            ausentes = t.update(person_id=1, epi_present_labels=[], required=[EpiTipo.HELMET],
                                 bbox_truncated=False)
            assert ausentes == []
        ausentes = t.update(person_id=1, epi_present_labels=[], required=[EpiTipo.HELMET],
                             bbox_truncated=False)
        assert ausentes == ["no_helmet"]

    def test_deteccao_no_meio_da_janela_reseta_a_contagem(self):
        # Cue central da pergunta do usuario: bbox some por 1 frame no meio
        # de uma sequencia, mas volta — nao pode contar como ausencia real.
        t = tracker(confirm_frames=3)
        t.update(person_id=1, epi_present_labels=[], required=[EpiTipo.HELMET],
                  bbox_truncated=False)
        t.update(person_id=1, epi_present_labels=[], required=[EpiTipo.HELMET],
                  bbox_truncated=False)
        # detectado de novo — reseta
        t.update(person_id=1, epi_present_labels=["HELMET"], required=[EpiTipo.HELMET],
                  bbox_truncated=False)
        t.update(person_id=1, epi_present_labels=[], required=[EpiTipo.HELMET],
                  bbox_truncated=False)
        ausentes = t.update(person_id=1, epi_present_labels=[], required=[EpiTipo.HELMET],
                             bbox_truncated=False)
        assert ausentes == []

    def test_detectado_todo_frame_nunca_confirma_ausencia(self):
        t = tracker(confirm_frames=2)
        for _ in range(10):
            ausentes = t.update(person_id=1, epi_present_labels=["HELMET"],
                                 required=[EpiTipo.HELMET], bbox_truncated=False)
            assert ausentes == []


class TestGuardas:
    def test_bbox_truncado_congela_a_contagem_em_vez_de_incrementar(self):
        # Item pode estar fora de quadro, nao ausente de verdade.
        t = tracker(confirm_frames=2)
        t.update(person_id=1, epi_present_labels=[], required=[EpiTipo.HELMET],
                  bbox_truncated=True)
        ausentes = t.update(person_id=1, epi_present_labels=[], required=[EpiTipo.HELMET],
                             bbox_truncated=True)
        assert ausentes == []

    def test_track_jovem_nao_confirma_ausencia(self):
        t = tracker(confirm_frames=1, min_track_frames=15)
        ausentes = []
        for _ in range(3):
            ausentes = t.update(person_id=1, epi_present_labels=[], required=[EpiTipo.HELMET],
                                 bbox_truncated=False)
        assert ausentes == []

    def test_track_maduro_confirma_normalmente(self):
        t = tracker(confirm_frames=1, min_track_frames=15)
        ausentes = []
        for _ in range(15):
            ausentes = t.update(person_id=1, epi_present_labels=[], required=[EpiTipo.HELMET],
                                 bbox_truncated=False)
        assert ausentes == ["no_helmet"]


class TestMultiplosItensEPessoas:
    def test_itens_diferentes_tem_contagens_independentes(self):
        t = tracker(confirm_frames=2)
        t.update(person_id=1, epi_present_labels=["HELMET"],
                  required=[EpiTipo.HELMET, EpiTipo.BOOTS], bbox_truncated=False)
        ausentes = t.update(person_id=1, epi_present_labels=["HELMET"],
                             required=[EpiTipo.HELMET, EpiTipo.BOOTS], bbox_truncated=False)
        assert ausentes == ["no_boots"]

    def test_pessoas_diferentes_tem_contagens_independentes(self):
        t = tracker(confirm_frames=1)
        ausentes_p1 = t.update(person_id=1, epi_present_labels=["HELMET"],
                                required=[EpiTipo.HELMET], bbox_truncated=False)
        ausentes_p2 = t.update(person_id=2, epi_present_labels=[],
                                required=[EpiTipo.HELMET], bbox_truncated=False)
        assert ausentes_p1 == []
        assert ausentes_p2 == ["no_helmet"]

    def test_item_nao_mais_exigido_nao_aparece_mesmo_se_ausente(self):
        t = tracker(confirm_frames=1)
        ausentes = t.update(person_id=1, epi_present_labels=[], required=[EpiTipo.VEST],
                             bbox_truncated=False)
        assert ausentes == ["no_vest"]


class TestForget:
    def test_forget_libera_o_estado_da_pessoa(self):
        t = tracker(confirm_frames=2)
        t.update(person_id=1, epi_present_labels=[], required=[EpiTipo.HELMET],
                  bbox_truncated=False)

        t.forget(1)

        # Sem forget, esta chamada confirmaria (streak chegaria a 2). Com
        # forget, o track "recomeca do zero" — precisa de novo das 2
        # ausencias consecutivas.
        ausentes = t.update(person_id=1, epi_present_labels=[], required=[EpiTipo.HELMET],
                             bbox_truncated=False)
        assert ausentes == []
