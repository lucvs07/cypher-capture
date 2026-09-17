"""
Testes de ZoneDetector.avaliacao_habilitada — o gate que decide se EPI/pose
sao avaliados para uma pessoa, a partir de zonas com delimita_avaliacao=True
(area de trabalho declarada; achado de campo SPI 2026-09-16).

Funcao pura, sem YOLO nem captura: so Zone/ZoneResult em memoria.
"""
from __future__ import annotations

from src.zones.detector import ZoneDetector, ZoneResult, ZoneStatus
from src.zones.manager import Zone


def zona(delimita_avaliacao: bool = False, zone_id: str = "z1") -> Zone:
    return Zone(
        zone_id=zone_id, name="Zona de teste", zone_type="restricted",
        polygon=[(0, 0), (10, 0), (10, 10)], delimita_avaliacao=delimita_avaliacao,
    )


def resultado(status: ZoneStatus, delimita_avaliacao: bool, zone_id: str = "z1") -> ZoneResult:
    return ZoneResult(
        person_id=1, zone_id=zone_id, zone_name="Zona de teste",
        status=status, distance_px=0.0, severity=3,
        delimita_avaliacao=delimita_avaliacao,
    )


class TestSemZonaDelimitadora:
    """Sem nenhuma zona delimitadora cadastrada, avalia todo mundo —
    comportamento anterior preservado."""

    def test_sem_zona_nenhuma_avalia_todo_mundo(self):
        assert ZoneDetector.avaliacao_habilitada([], []) is True

    def test_so_zona_de_perigo_simples_nao_restringe_nada(self):
        zonas = [zona(delimita_avaliacao=False)]
        # pessoa nem sequer aparece nos resultados (fora do buffer de aproximacao)
        assert ZoneDetector.avaliacao_habilitada([], zonas) is True

    def test_zona_de_perigo_simples_com_pessoa_dentro_ainda_avalia(self):
        zonas = [zona(delimita_avaliacao=False)]
        resultados = [resultado(ZoneStatus.INSIDE, delimita_avaliacao=False)]
        assert ZoneDetector.avaliacao_habilitada(resultados, zonas) is True


class TestComZonaDelimitadora:
    """Existindo ao menos uma zona delimitadora, so quem esta dentro dela
    e avaliado."""

    def test_zona_delimitadora_cadastrada_pessoa_fora_nao_avalia(self):
        zonas = [zona(delimita_avaliacao=True)]
        assert ZoneDetector.avaliacao_habilitada([], zonas) is False

    def test_zona_delimitadora_cadastrada_pessoa_dentro_avalia(self):
        zonas = [zona(delimita_avaliacao=True)]
        resultados = [resultado(ZoneStatus.INSIDE, delimita_avaliacao=True)]
        assert ZoneDetector.avaliacao_habilitada(resultados, zonas) is True

    def test_zona_delimitadora_pessoa_so_aproximando_nao_avalia(self):
        """APPROACHING nao e INSIDE — so entrar de fato habilita a avaliacao."""
        zonas = [zona(delimita_avaliacao=True)]
        resultados = [resultado(ZoneStatus.APPROACHING, delimita_avaliacao=True)]
        assert ZoneDetector.avaliacao_habilitada(resultados, zonas) is False

    def test_pessoa_dentro_de_zona_de_perigo_mas_fora_da_delimitadora_nao_avalia(self):
        """Camera com as duas zonas: perigo (nao restringe) + area de
        trabalho (restringe). Estar dentro da zona de perigo sozinha nao
        basta — precisa estar dentro da que delimita avaliacao."""
        zonas = [zona(delimita_avaliacao=False, zone_id="perigo"),
                 zona(delimita_avaliacao=True, zone_id="trabalho")]
        resultados = [resultado(ZoneStatus.INSIDE, delimita_avaliacao=False, zone_id="perigo")]
        assert ZoneDetector.avaliacao_habilitada(resultados, zonas) is False
