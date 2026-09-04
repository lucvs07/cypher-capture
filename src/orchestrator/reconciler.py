"""
Ciclo de reconciliação do orquestrador (design doc, §2.3).

A cada chamada de `reconciliar_uma_vez()`: busca o catálogo desejado,
encerra o que saiu dele, detecta mudança de streamUrl e processo morto
(com backoff exponencial — mesma fórmula de `ResilientCapture`), e sobe o
que falta respeitando o limite de concorrência. Nunca levanta exceção para
falhas esperadas (catálogo inacessível vira "não faz nada neste ciclo").
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from src.orchestrator.registry import ProcessRegistry
from src.orchestrator.session import CatalogoInacessivelError

log = logging.getLogger(__name__)


def _agora_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _EstadoDeFalha:
    espera_atual_segundos: float
    proxima_tentativa_permitida: datetime


class Reconciler:
    def __init__(self, sessao, registry: ProcessRegistry, api_url: str,
                 max_concorrentes: int, backoff_base: float, backoff_max: float,
                 agora: Callable[[], datetime] = _agora_utc) -> None:
        self.sessao = sessao
        self.registry = registry
        self.api_url = api_url
        self.max_concorrentes = max_concorrentes
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.agora = agora
        self._falhas: dict[str, _EstadoDeFalha] = {}

    def reconciliar_uma_vez(self) -> None:
        try:
            desejadas = self.sessao.listar_cameras()
        except CatalogoInacessivelError:
            return  # fail-safe: nao mexe em nenhum processo neste ciclo

        desejado_por_id = {c.camera_id: c for c in desejadas}

        # 2. encerra o que saiu do catalogo (desativada ou deletada)
        for camera_id in list(self.registry.camera_ids_rastreados()):
            if camera_id not in desejado_por_id:
                self.registry.encerrar(camera_id)
                self._falhas.pop(camera_id, None)

        precisa_reiniciar: list[str] = []  # ja rodava antes deste ciclo (config mudou / morreu sozinha)
        precisa_iniciar_novo: list[str] = []  # nunca rastreada -- camera nova no catalogo

        for camera_id, camera in desejado_por_id.items():
            viva = self.registry.esta_vivo(camera_id)
            rastreada = camera_id in self.registry.camera_ids_rastreados()

            if viva and self.registry.stream_url_atual(camera_id) != camera.stream_url:
                # 3. mudou de config -- encerra e marca para reiniciar com a nova
                self.registry.encerrar(camera_id)
                self._falhas.pop(camera_id, None)
                precisa_reiniciar.append(camera_id)
                continue

            if viva:
                self._falhas.pop(camera_id, None)  # rodando com sucesso -- reseta o backoff
                continue

            if rastreada:
                # 4. estava rastreada mas nao esta mais viva -- morreu sozinha
                if not self._pode_reiniciar_agora(camera_id):
                    continue
                self.registry.encerrar(camera_id)  # limpa a entrada morta antes de reiniciar
                precisa_reiniciar.append(camera_id)
                continue

            precisa_iniciar_novo.append(camera_id)

        # 5. sobe o que falta, respeitando o limite de concorrencia -- cameras
        # que ja rodavam (config mudou ou morreram sozinhas) tem prioridade
        # sobre cameras novas, senao uma camera nova alfabeticamente menor
        # pode roubar a vaga de uma que so precisa ser reiniciada, deixando-a
        # sem vaga indefinidamente enquanto a nova continuar viva.
        vivos = sum(1 for cid in self.registry.camera_ids_rastreados() if self.registry.esta_vivo(cid))
        vagas = max(self.max_concorrentes - vivos, 0)
        candidatos = sorted(precisa_reiniciar) + sorted(precisa_iniciar_novo)
        iniciadas = candidatos[:vagas]
        for camera_id in iniciadas:
            camera = desejado_por_id[camera_id]
            self.registry.iniciar(camera_id, camera.stream_url, self.api_url)

        # 6. loga um resumo do ciclo
        rodando = sum(1 for cid in self.registry.camera_ids_rastreados() if self.registry.esta_vivo(cid))
        na_fila = len(candidatos) - len(iniciadas)
        log.info("Ciclo de reconciliacao: rodando=%d na_fila=%d reiniciadas=%d",
                 rodando, na_fila, len(iniciadas))

    def _pode_reiniciar_agora(self, camera_id: str) -> bool:
        """Mesma formula de backoff de ResilientCapture._loop_leitura: a
        primeira falha ja espera backoff_base (nao tenta de novo na hora), e
        cada falha subsequente dobra a espera ate backoff_max."""
        estado = self._falhas.get(camera_id)
        if estado is None:
            self._falhas[camera_id] = _EstadoDeFalha(
                espera_atual_segundos=self.backoff_base,
                proxima_tentativa_permitida=self.agora() + timedelta(seconds=self.backoff_base))
            return False
        if self.agora() < estado.proxima_tentativa_permitida:
            return False
        proxima_espera = min(estado.espera_atual_segundos * 2, self.backoff_max)
        self._falhas[camera_id] = _EstadoDeFalha(
            espera_atual_segundos=proxima_espera,
            proxima_tentativa_permitida=self.agora() + timedelta(seconds=proxima_espera))
        return True
