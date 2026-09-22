"""
Liberacao do estado por pessoa de tracks que sumiram do quadro.

Classificador, analise temporal, suavizador de REBA e inferencia de ausencia
de EPI guardam estado por ``person_id``. O ByteTrack da id novo a cada pessoa
que entra no quadro, entao sem isto esses dicionarios so crescem ao longo de
uma execucao que roda por dias.
"""
from __future__ import annotations

from typing import Callable, Iterable


class EsquecedorDeTracks:
    """Chama ``forget(person_id)`` nos donos de estado quando um track fica
    ``apos_frames`` sem aparecer.

    ``apos_frames`` precisa ser maior que o ``track_buffer`` do ByteTrack:
    dentro dele o track ainda pode voltar com o mesmo id, e esquecer antes
    zeraria a histerese e a contagem de ausencia de EPI de uma pessoa que so
    ficou ocluida por um instante.
    """

    def __init__(self, apos_frames: int, alvos: Iterable[Callable[[int], None]]) -> None:
        self.apos_frames = apos_frames
        self._alvos = list(alvos)
        self._visto_em: dict[int, int] = {}

    def update(self, frame_idx: int, ids_vistos: Iterable[int]) -> list[int]:
        """Registra os ids do frame e devolve os que acabaram de ser esquecidos."""
        for person_id in ids_vistos:
            self._visto_em[person_id] = frame_idx

        limite = frame_idx - self.apos_frames
        sumidos = [pid for pid, visto in self._visto_em.items() if visto <= limite]
        for person_id in sumidos:
            del self._visto_em[person_id]
            for forget in self._alvos:
                forget(person_id)
        return sumidos
