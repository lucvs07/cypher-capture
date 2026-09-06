#!/usr/bin/env python3
"""
Regrava tests/fixtures/cenario2_transitions.json rodando o PostureClassifier
atualizado (gating temporal de queda, design doc 2026-09-05) contra o vídeo
real. Roda toda a pilha de pose+classificação frame a frame via
UnifiedPipeline.process_frame — não é um teste automatizado (precisa do
vídeo e do modelo YOLO), é um utilitário de manutenção da fixture.

Uso:
    python scripts/regen_cenario2_fixture.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from src.fusion.pipeline import PipelineConfig, UnifiedPipeline

VIDEO = "data/validation/cenario2.mp4"
FIXTURE = Path("tests/fixtures/cenario2_transitions.json")

# Fatos do vídeo, anotados à mão — não mudam com o classificador.
QUEDAS_REAIS = json.loads(FIXTURE.read_text())["quedas_reais"]


def main() -> None:
    config = PipelineConfig.from_yaml("config/pose_config.yaml")
    pipeline = UnifiedPipeline(config)

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)

    transitions: list[dict] = []
    frame_idx = -1
    vida_dos_tracks: dict[int, int] = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        _, events = pipeline.process_frame(frame)
        for event in events:
            vida_dos_tracks[event.person_id] += 1
            if event.posture != event.previous_posture:
                transitions.append({
                    "frame": frame_idx,
                    "track": event.person_id,
                    "from": event.previous_posture,
                    "to": event.posture,
                })
    cap.release()

    dados = {
        "video": VIDEO,
        "total_frames": frame_idx + 1,
        "fps": fps,
        "quedas_reais": QUEDAS_REAIS,
        "transitions": transitions,
        "vida_dos_tracks": {str(k): v for k, v in vida_dos_tracks.items()},
    }
    FIXTURE.write_text(json.dumps(dados, indent=2, ensure_ascii=False))
    print(f"Fixture regravada: {len(transitions)} transicoes, "
          f"{len(vida_dos_tracks)} track(s), {frame_idx + 1} frames.")


if __name__ == "__main__":
    main()
