"""
Liberacao do estado por pessoa (src/fusion/track_state.py + forget() dos
modulos com estado).

Achado 2026-09-22: EpiAbsenceTracker.forget() existia mas nunca era chamado,
e classificador, analise temporal e suavizador de REBA nem tinham forget —
todo id que o ByteTrack ja deu ficava na memoria ate o fim da execucao.
"""
from src.fusion.track_state import EsquecedorDeTracks
from src.pose.angles import JointAngles
from src.pose.classifier import PostureClassifier
from src.pose.epi_absence import EpiAbsenceTracker
from src.pose.epi_catalog import EpiTipo
from src.pose.reba import REBAScore
from src.pose.reba_smoother import RebaSmoother
from src.pose.temporal import TemporalAnalyzer


class TestEsquecedorDeTracks:
    def test_esquece_track_que_sumiu_por_mais_que_o_limite(self):
        esquecidos = []
        esquecedor = EsquecedorDeTracks(apos_frames=10, alvos=[esquecidos.append])
        esquecedor.update(1, [7])

        for frame in range(2, 12):
            esquecedor.update(frame, [])

        assert esquecidos == [7]

    def test_nao_esquece_dentro_do_limite(self):
        # Oclusao breve: o ByteTrack ainda devolve o mesmo id dentro do
        # track_buffer, e esquecer ai zeraria histerese e contagem de EPI.
        esquecidos = []
        esquecedor = EsquecedorDeTracks(apos_frames=10, alvos=[esquecidos.append])
        esquecedor.update(1, [7])

        for frame in range(2, 11):
            esquecedor.update(frame, [])
        esquecedor.update(11, [7])

        assert esquecidos == []

    def test_track_visto_de_novo_reinicia_a_contagem(self):
        esquecidos = []
        esquecedor = EsquecedorDeTracks(apos_frames=5, alvos=[esquecidos.append])
        esquecedor.update(1, [7])
        esquecedor.update(4, [7])

        esquecedor.update(8, [])
        assert esquecidos == []
        esquecedor.update(9, [])
        assert esquecidos == [7]

    def test_chama_todos_os_alvos_uma_vez_so(self):
        a, b = [], []
        esquecedor = EsquecedorDeTracks(apos_frames=1, alvos=[a.append, b.append])
        esquecedor.update(1, [3])

        esquecedor.update(2, [])
        esquecedor.update(3, [])

        assert a == [3] and b == [3]


def test_classifier_forget_libera_estado():
    classifier = PostureClassifier()
    classifier.update(1, JointAngles(person_id=1), REBAScore(person_id=1),
                      (0, 0, 100, 200), False)
    assert 1 in classifier._states

    classifier.forget(1)

    assert 1 not in classifier._states
    assert 1 not in classifier._pending
    assert 1 not in classifier._exit_counters
    assert 1 not in classifier._fall_pending_confirm


def test_temporal_forget_libera_estado():
    temporal = TemporalAnalyzer()
    temporal.update(1, JointAngles(person_id=1, trunk_flexion=10.0), (0, 0, 100, 200),
                    is_bad_posture=True)
    assert 1 in temporal._com

    temporal.forget(1)

    assert 1 not in temporal._com
    assert 1 not in temporal._trunk_angles
    assert 1 not in temporal._bad_posture_streak


def test_reba_smoother_forget_libera_estado():
    smoother = RebaSmoother()
    smoother.smooth(1, REBAScore(person_id=1, final_score=5))
    assert 1 in smoother._history

    smoother.forget(1)

    assert 1 not in smoother._history
    assert 1 not in smoother._last_valid


def test_epi_absence_forget_libera_estado():
    tracker = EpiAbsenceTracker(confirm_frames=3, min_track_frames=1)
    tracker.update(1, [], [EpiTipo.HELMET], bbox_truncated=False)
    assert tracker._frames_seen

    tracker.forget(1)

    assert tracker._frames_seen == {}
    assert tracker._streak == {}
