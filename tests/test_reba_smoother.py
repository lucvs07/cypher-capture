"""
Testes de RebaSmoother (src/pose/reba_smoother.py) - suaviza final_score por
mediana movel, pulando frames indeterminados (achado da auditoria: REBA era
calculado cru por frame, sem nenhuma suavizacao, so a categoria discreta
tinha histerese).
"""
from __future__ import annotations

from src.pose.reba import REBAScore
from src.pose.reba_smoother import RebaSmoother


def score(final: int, indeterminate: bool = False) -> REBAScore:
    return REBAScore(person_id=1, final_score=final, indeterminate=indeterminate)


def test_primeiro_frame_devolve_o_proprio_valor():
    smoother = RebaSmoother(window=5)
    assert smoother.smooth(1, score(4)) == 4


def test_suaviza_por_mediana_da_janela():
    smoother = RebaSmoother(window=3)
    smoother.smooth(1, score(2))
    smoother.smooth(1, score(2))
    resultado = smoother.smooth(1, score(9))  # outlier isolado
    assert resultado == 2  # mediana de [2, 2, 9]


def test_frame_indeterminado_nao_entra_na_janela_e_devolve_ultimo_valido():
    smoother = RebaSmoother(window=3)
    smoother.smooth(1, score(7))
    resultado = smoother.smooth(1, score(1, indeterminate=True))
    assert resultado == 7  # nao contamina com o score(1) de fallback


def test_pessoas_diferentes_tem_janelas_independentes():
    smoother = RebaSmoother(window=3)
    smoother.smooth(1, score(9))
    resultado_pid2 = smoother.smooth(2, score(2))
    assert resultado_pid2 == 2
