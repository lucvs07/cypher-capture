"""
Testes do ciclo de vida do incidente (src/notify/).

O que está sendo protegido: a tradução de transições da máquina de estados em
episódios com início e fim. Sem janela de merge, as 2 quedas reais de
cenario2.mp4 produzem 10 incidentes; sem eviction, 2 incidentes ficam abertos
para sempre. Ambos os números foram medidos, não estimados — ver a spec §4.1.2.

Nada aqui exige GPU ou modelo: os eventos são construídos à mão.
"""
from __future__ import annotations

from src.fusion.engine import PersonEvent


def test_evento_nasce_com_postura_anterior_normal():
    evento = PersonEvent(person_id=1)

    assert evento.previous_posture == "normal"


def test_postura_anterior_e_um_campo_independente_da_atual():
    evento = PersonEvent(person_id=1, posture="immediate_risk",
                         previous_posture="normal")

    assert evento.posture == "immediate_risk"
    assert evento.previous_posture == "normal"
