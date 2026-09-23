"""
Inferencia de ausencia de EPI por debounce temporal.

O detector so produz classes de presenca (achado 2026-09-20 — ver
epi_catalog.py); "ausente" e sempre uma inferencia sobre a HISTORIA de uma
pessoa rastreada, nunca um dado direto de 1 frame. A confusao a evitar: bbox
que nao aparece num frame (oclusao, blur, angulo momentaneo) nao e a mesma
coisa que o item ter sido removido — so um item que fica fora de deteccao por
uma janela inteira de frames consecutivos e reportado como ausente. A
deteccao do item em QUALQUER frame no meio da janela reseta a contagem, ainda
que a janela de confirmacao ja tenha andado bastante.
"""
from __future__ import annotations

from .epi_catalog import EpiTipo


class EpiAbsenceTracker:
    def __init__(self, confirm_frames: int, min_track_frames: int = 15) -> None:
        self.confirm_frames = confirm_frames
        self.min_track_frames = min_track_frames
        # (person_id, EpiTipo) -> frames consecutivos sem deteccao
        self._streak: dict[tuple[int, EpiTipo], int] = {}
        # Idade do track, contada aqui (nao recebida de fora): cada
        # maquina de estados por frame no pipeline mantem a propria
        # contagem, mesmo padrao de IncidentTracker._frames_seen.
        self._frames_seen: dict[int, int] = {}

    def update(
        self,
        person_id: int,
        epi_present: list[EpiTipo],
        required: list[EpiTipo],
        bbox_truncated: bool,
    ) -> list[EpiTipo]:
        """Devolve os EPIs exigidos confirmados como ausentes para esta
        pessoa neste frame."""
        frames_seen = self._frames_seen.get(person_id, 0) + 1
        self._frames_seen[person_id] = frames_seen
        if frames_seen < self.min_track_frames:
            return []

        presentes = set(epi_present)

        confirmados: list[EpiTipo] = []
        for tipo in required:
            chave = (person_id, tipo)
            if tipo in presentes:
                self._streak.pop(chave, None)
                continue

            if bbox_truncated:
                # Item pode estar fora de quadro, nao ausente de verdade —
                # congela a contagem em vez de incrementar ou resetar.
                continue

            streak = self._streak.get(chave, 0) + 1
            self._streak[chave] = streak
            if streak >= self.confirm_frames:
                confirmados.append(tipo)

        return confirmados

    def forget(self, person_id: int) -> None:
        """Libera o estado de uma pessoa cujo track terminou (saiu de quadro
        definitivamente, nao só um frame perdido) — evita crescimento sem
        fim do dicionario ao longo de uma execucao longa."""
        self._frames_seen.pop(person_id, None)
        for chave in [k for k in self._streak if k[0] == person_id]:
            del self._streak[chave]
