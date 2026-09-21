from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal


@dataclass
class Zone:
    zone_id: str
    name: str
    zone_type: Literal["restricted", "attention"]
    polygon: list[tuple[float, float]]   # ordered (x, y) vertices in pixel coords
    severity: int = 3                    # 1=low … 4=critical
    required_ppe: list[str] = field(default_factory=list)
    # Achado SPI 2026-09-16: ate entao zona so influenciava severidade de
    # alerta, nunca o que era avaliado (EPI/pose sempre rodavam pra todo
    # mundo). Quando True, esta zona e a area de trabalho: EPI e pose so
    # sao avaliados pra quem esta dentro dela. Default False preserva o
    # comportamento anterior.
    delimita_avaliacao: bool = False

    @staticmethod
    def new(
        name: str,
        zone_type: Literal["restricted", "attention"],
        polygon: list[tuple[float, float]],
        severity: int = 3,
        required_ppe: list[str] | None = None,
        delimita_avaliacao: bool = False,
    ) -> "Zone":
        return Zone(
            zone_id=str(uuid.uuid4())[:8],
            name=name,
            zone_type=zone_type,
            polygon=polygon,
            severity=severity,
            required_ppe=required_ppe or [],
            delimita_avaliacao=delimita_avaliacao,
        )


class ZoneManager:
    """Manages critical zone definitions with JSON persistence.

    Aceita zonas ja resolvidas em memoria (``zones=``), usadas quando o
    cliente remoto de configuracao (``ZoneConfigClient``, design doc de
    zonas de risco §3.1) ja converteu normalizado->pixel com a resolucao
    real do frame — nesse caso nada e lido nem gravado em disco.
    """

    def __init__(self, config_path: str = "config/zones/zones.json",
                 zones: list[Zone] | None = None) -> None:
        self._path = Path(config_path)
        self._zones: dict[str, Zone] = {}
        if zones is not None:
            for z in zones:
                self._zones[z.zone_id] = z
        elif self._path.exists():
            self._load()

    def add(self, zone: Zone) -> None:
        self._zones[zone.zone_id] = zone
        self._save()

    def remove(self, zone_id: str) -> None:
        self._zones.pop(zone_id, None)
        self._save()

    def get(self, zone_id: str) -> Zone | None:
        return self._zones.get(zone_id)

    def get_all(self) -> list[Zone]:
        return list(self._zones.values())

    def clear(self) -> None:
        self._zones.clear()
        self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump([asdict(z) for z in self._zones.values()], f, indent=2)

    def _load(self) -> None:
        with open(self._path) as f:
            data = json.load(f)
        # Ignore unknown keys so a hand-annotated zones.json (e.g. with a
        # "comment" field) can't crash the pipeline at startup.
        known = {f.name for f in fields(Zone)}
        for item in data:
            item = {k: v for k, v in item.items() if k in known}
            item["polygon"] = [tuple(p) for p in item["polygon"]]
            z = Zone(**item)
            self._zones[z.zone_id] = z
