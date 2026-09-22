"""
Caracterizacao do fluxo de EPI com os pesos reais (detector + pose).

Roda o ``UnifiedPipeline`` de verdade sobre um video de ``data/validation/``
e grava, por frame e por track, so dados discretos: EPI associado a pessoa,
EPI confirmado como ausente, severidade e abertura/fechamento de incidente.
Nada de confianca nem coordenada de bbox — o que tem que ficar identico entre
duas versoes do codigo e a decisao, nao o float.

Existe pra provar que mudar a representacao do EPI (label cru do detector /
``no_*`` -> ``EpiTipo``) nao muda o comportamento: o golden e gravado ANTES da
refatoracao e ``test_epi_flow_golden.py`` exige resultado identico depois.

Regravar (so quando a mudanca de comportamento for intencional):
    venv/bin/python tests/epi_flow_golden.py --record
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GOLDEN_DIR = ROOT / "tests" / "fixtures" / "epi_flow"

# Pesos explicitos (nao o default do PipelineConfig): o golden nao pode mudar
# so porque o padrao do projeto trocou de modelo.
EPI_WEIGHTS = "models/exp-m4-1709-v4.pt"
POSE_WEIGHTS = "yolo11s-pose.pt"
DEVICE = "cuda"

# Area de trabalho = quadro inteiro, exigindo 4 itens: cobre EPI presente,
# ausencia confirmada e o +1 da zona (taxonomia §4.2) no mesmo cenario.
REQUIRED_PPE = ["HELMET", "GLOVE", "BOOTS", "VEST"]

SCENARIOS = {
    "cenario4": "data/validation/cenario4.mp4",
    "test_bus": "data/validation/test_bus.mp4",
}

# Formatos legados que o fluxo usava antes do EpiTipo trafegar direto —
# mantidos aqui so pra o gravador ler as duas versoes do codigo.
_LEGADO = {
    "HELMET": "HELMET", "Goggles": "GOGGLES", "GLOVE": "GLOVE",
    "VEST": "VEST", "boots": "BOOTS", "MASK": "MASK",
    "no_helmet": "HELMET", "no_goggle": "GOGGLES", "no_gloves": "GLOVE",
    "no_vest": "VEST", "no_boots": "BOOTS", "no_mask": "MASK",
}


def _epi(valor) -> str:
    nome = getattr(valor, "value", valor)
    return _LEGADO.get(nome, nome)


def _epis(valores) -> list[str]:
    return sorted({_epi(v) for v in valores})


def _zone_file(tmp_dir: Path, width: int, height: int) -> Path:
    path = tmp_dir / "zones.json"
    path.write_text(json.dumps([{
        "zone_id": "golden01",
        "name": "Area de trabalho",
        "zone_type": "attention",
        "polygon": [[0, 0], [width, 0], [width, height], [0, height]],
        "severity": 2,
        "required_ppe": REQUIRED_PPE,
        "delimita_avaliacao": True,
    }]))
    return path


def run_scenario(video: str, tmp_dir: Path) -> dict:
    from src.fusion.pipeline import PipelineConfig, UnifiedPipeline
    from src.notify.tracker import IncidentTracker, TrackerConfig

    cap = cv2.VideoCapture(str(ROOT / video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    config = PipelineConfig(
        epi_model_path=str(ROOT / EPI_WEIGHTS),
        pose_model_path=str(ROOT / POSE_WEIGHTS),
        pose_tracker_config=str(ROOT / "config" / "bytetrack_tuned.yaml"),
        device=DEVICE,
        zones_config=str(_zone_file(tmp_dir, width, height)),
        # Stride adaptativo depende de latencia de relogio — desligado pra
        # o EPI rodar em todo frame e a execucao ser reproduzivel.
        max_latency_ms=1e9,
    )
    pipeline = UnifiedPipeline(config)

    incidents: list[dict] = []

    def registrar(acao):
        def handler(rep):
            incidents.append({
                "frame": frame_idx, "action": acao, "track": rep.track_id,
                "type": rep.type, "severity": rep.severity,
                "peak_severity": rep.peak_severity,
                "epi_present": _epis(rep.epi_present),
                "epi_missing": _epis(rep.epi_missing),
                "closed_reason": rep.closed_reason,
            })
        return handler

    tracker = IncidentTracker(
        TrackerConfig(camera_id="GOLDEN", run_id="golden", fps=config.target_fps),
        on_open=registrar("open"), on_close=registrar("close"),
    )

    # Relogio sintetico: o tracker so usa `now` pra merge/renotify, e tempo
    # de parede tornaria o golden dependente da velocidade da GPU.
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    frames: list[dict] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            _, events = pipeline.process_frame(frame)
            tracker.process(events, t0 + timedelta(seconds=frame_idx / config.target_fps))
            frames.append({
                "frame": frame_idx,
                "people": [
                    {
                        "track": ev.person_id,
                        "epi_present": _epis(ev.epi_present),
                        "epi_missing": _epis(ev.epi_violations),
                        "severity": ev.severity,
                        "zone_status": ev.zone_status,
                    }
                    for ev in sorted(events, key=lambda e: e.person_id)
                ],
            })
    finally:
        cap.release()
    tracker.flush(t0 + timedelta(seconds=(frame_idx + 1) / config.target_fps))

    return {"video": video, "required_ppe": REQUIRED_PPE, "epi_weights": EPI_WEIGHTS,
            "frames": frames, "incidents": incidents}


def summarize(result: dict) -> dict:
    people = [p for f in result["frames"] for p in f["people"]]
    return {
        "frames": len(result["frames"]),
        "person_frames": len(people),
        "with_epi_present": sum(1 for p in people if p["epi_present"]),
        "with_epi_missing": sum(1 for p in people if p["epi_missing"]),
        "escalated": sum(1 for p in people if p["epi_missing"] and p["severity"] >= 3),
        "epi_incidents": sum(1 for i in result["incidents"]
                             if i["type"] == "EPI_VIOLATION" and i["action"] == "open"),
    }


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def main() -> None:
    import tempfile

    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="grava o golden")
    parser.add_argument("--out", type=Path, help="grava em outro diretorio (comparacao)")
    args = parser.parse_args()

    destino = args.out or (GOLDEN_DIR if args.record else None)
    for name, video in SCENARIOS.items():
        with tempfile.TemporaryDirectory() as tmp:
            result = run_scenario(video, Path(tmp))
        print(name, summarize(result))
        if destino is not None:
            destino.mkdir(parents=True, exist_ok=True)
            (destino / f"{name}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
