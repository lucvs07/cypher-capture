# Plano de Testes — Dia 1 (23/06/2026)

**Objetivo do dia (6h):** ao final, ter (a) o CLI do pipeline unificado funcionando, (b) os **3 vídeos gravados**, e (c) a **latência E2E medida**. Esses são os bloqueadores do Dia 2.
**Mentoria SPI:** quinta-feira 25/06/2026 — o Dia 1 produz a evidência que sustenta o checkpoint.

> **Artefatos já criados nesta sessão** (compilam, `py_compile` OK):
> - `src/run_pipeline.py` — CLI do `UnifiedPipeline` (novo)
> - `scripts/annotate_gt.py` — anotador de ground-truth (novo)
> - `config/zones/zones.example.json` — zona de exemplo p/ Cenário 3 (novo)
> - `src/fusion/pipeline.py` — instrumentado com `self.last_timings` (editado)

---

## Status pré-execução (checklist 2d)

Estado **real** do repositório hoje (verificado no ambiente):

| Item | Estado atual | Ação Dia 1 |
|---|---|---|
| **Ambiente Python / deps** | ❌ **Não há venv; `ultralytics`/`torch`/`cv2` não instalados.** `python` nem existe (só `python3` 3.10). | **Passo 0 obrigatório:** criar venv + `pip install -r requirements.txt`. |
| `yolo11s-pose.pt` baixado? | ❌ Ausente. Ultralytics baixa **na 1ª execução**, para o **CWD** (raiz do projeto). | Pré-baixar (passo 0) **antes** de gravar, p/ não dar timeout no meio do teste. |
| `config/zones/zones.json` existe? | ❌ Só `.gitkeep`. **Pipeline NÃO quebra sem ele** (`ZoneManager` carrega lista vazia). | Necessário só p/ Cenário 3 → criar via `annotate_zones.py` ou copiar o exemplo. |
| `pose_config.yaml` device | `device: cpu` | Conferir se há GPU (`torch.cuda.is_available()`); decidir cpu/cuda. |
| Modelo EPI `exp.pt` — classes | ⚠️ **Discrepância não resolvida.** Você descreve 7 classes (GLOVE, Goggles, HELMET, MASK, Person, VEST, boots — **sem classes `no_*`**), mas `engine.py` casa violações com `{no_gloves,no_goggle,no_helmet,no_boots,none}`. Não consegui introspectar (deps ausentes). | **Verificar 1ª coisa:** `python -c "from ultralytics import YOLO; print(YOLO('models/exp.pt').names)"`. Se não houver `no_*`, **o cruzamento EPI+pose não dispara** (ver Parte 3, problema #5). |
| Iluminação p/ keypoints conf > 0,5 | — | Garantir luz frontal difusa; sem contraluz. |
| Câmera cobre corpo inteiro | — | **Tornozelos visíveis** (zonas) e **quadril/joelho** (ângulos). Ver framing por cenário em 2a. |
| `run_pipeline.py` testado com fonte real | ❌ ainda não | Testar com webcam **antes** de gravar (passo 2). |
| Espaço em disco p/ vídeos anotados | — | Reservar ~1–2 GB. |

---

## Passo 0 — Setup do ambiente (fazer 1× no início, ~20 min)

```bash
cd cypher-capture
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pré-baixar os dois modelos de pose para a RAIZ do projeto (CWD).
# Rode SEMPRE a partir da raiz — os caminhos no YAML são relativos a ela.
python -c "from ultralytics import YOLO; YOLO('yolo11s-pose.pt'); YOLO('yolo11n-pose.pt'); print('pose models OK')"

# Verificação crítica das classes do detector de EPI (resolve a discrepância 7 vs 11):
python -c "from ultralytics import YOLO; m=YOLO('models/exp.pt'); print('task:',m.task,'| names:',m.names)"

# GPU disponível?
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

Dentro do venv `python` já aponta para o 3.10 — daqui em diante os comandos usam `python`.

---

## Parte 1 — CLI do UnifiedPipeline

### Decisão de design

`UnifiedPipeline.run_video` **não serve** para o Dia 1: não salva output, não roda headless de forma configurável, não imprime latência e **descarta os `events`** (`pipeline.py` linha ~188). Em vez de reescrevê-lo, `src/run_pipeline.py` conduz o loop ele mesmo chamando `process_frame`, e adiciona: gravação de vídeo, modo headless, HUD de latência por etapa e um *sink* de eventos (impressão no terminal). `run_video` fica intacto.

Para isso, `process_frame` foi instrumentado (Parte 3, problema #2): agora popula `self.last_timings` com `pose_ms / epi_ms / epi_ran / analysis_ms / overlay_ms / total_ms`.

### Comandos de execução

```bash
# Webcam, headless (medição em máquina sem GUI). Para 5s ~ usar --max-frames.
python src/run_pipeline.py --source 0 --no-display

# Webcam com janela (validar visualmente antes de gravar)
python src/run_pipeline.py --source 0

# Vídeo de teste, salva output anotado + janela
python src/run_pipeline.py \
  --source data/validation/cenario1.mp4 \
  --output outputs/captures/cenario1_annot.mp4

# Com zonas explícitas (Cenário 3)
python src/run_pipeline.py --source data/validation/cenario3.mp4 \
  --zones config/zones/zones.json \
  --output outputs/captures/cenario3_annot.mp4
```

**Output esperado no terminal** (atualiza em `\r` a cada frame):
```
[config] device=cpu  pose=yolo11s-pose.pt  epi=models/exp.pt
[config] zones=config/zones/zones.json  target_fps=15.0  budget=200ms
[zones]  1 zone(s) loaded
f   42   118.3ms (avg 121.7 |  8.2 FPS)  pose  74.1 | epi  31.0 | analysis  6.2 | overlay  4.8 | stride 1
[EVENT] f42 id#1 sev=2 posture=ergonomically_inadequate reba=4(AL3) zone=outside/- epi=- :: prolonged_squat
...
============================================================
  Frames         : 300
  E2E latency    : mean=121.7ms  p50=118.0ms  p95=152.3ms  max=410.1ms
  Latency target : PASS ✓  (p95 ≤ 200ms)
============================================================
```

### `src/run_pipeline.py` (completo)

> Já gravado no repo. Flags: `--source` (obrigatório), `--config`, `--zones`, `--output`, `--no-display`, `--min-severity` (default 2), `--max-frames` (default 0 = ilimitado).

```python
#!/usr/bin/env python3
"""CLI entry-point for the unified EPI + pose pipeline (Phase 2)."""
from __future__ import annotations
import argparse, sys, time
from collections import deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2, numpy as np
from src.fusion.pipeline import PipelineConfig, UnifiedPipeline
from src.utils.video import open_capture

def parse_args():
    p = argparse.ArgumentParser(description="Cypher — unified EPI + pose pipeline")
    p.add_argument("--source", required=True, help="webcam index | video path | RTSP URL")
    p.add_argument("--config", default="config/pose_config.yaml")
    p.add_argument("--zones", default=None, help="override zones JSON (default: from config)")
    p.add_argument("--output", default=None, help="save annotated video (e.g. out.mp4)")
    p.add_argument("--no-display", action="store_true", help="run headless (no GUI window)")
    p.add_argument("--min-severity", type=int, default=2, help="print events at/above this severity")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = unlimited)")
    return p.parse_args()

def main():
    args = parse_args()
    config = PipelineConfig.from_yaml(args.config)
    if args.zones:
        config.zones_config = args.zones
    print(f"[config] device={config.device}  pose={config.pose_model_path}  epi={config.epi_model_path}")
    print(f"[config] zones={config.zones_config}  target_fps={config.target_fps}  budget={config.max_latency_ms:.0f}ms")
    pipeline = UnifiedPipeline(config)
    print(f"[zones]  {len(pipeline._zones.get_all())} zone(s) loaded")
    cap = open_capture(args.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    src_fps = cap.get(cv2.CAP_PROP_FPS) or config.target_fps
    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height))
    window = "Cypher — EPI + Pose (q to quit)"
    if not args.no_display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    roll, all_lat, frame_idx = deque(maxlen=30), [], 0
    t_start = time.perf_counter()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.perf_counter()
            annotated, events = pipeline.process_frame(frame)
            dt_ms = (time.perf_counter() - t0) * 1000
            roll.append(dt_ms); all_lat.append(dt_ms); frame_idx += 1
            tm = pipeline.last_timings
            avg = sum(roll) / len(roll); fps = 1000.0 / avg if avg > 0 else 0.0
            epi_tag = "" if tm.get("epi_ran") else "(skip)"
            print(f"f{frame_idx:5d}  {dt_ms:6.1f}ms (avg {avg:5.1f} | {fps:4.1f} FPS)  "
                  f"pose {tm.get('pose_ms',0):5.1f} | epi {tm.get('epi_ms',0):5.1f}{epi_tag} | "
                  f"analysis {tm.get('analysis_ms',0):4.1f} | overlay {tm.get('overlay_ms',0):4.1f} | "
                  f"stride {pipeline._epi_stride}", end="\r", flush=True)
            for ev in events:
                if ev.severity >= args.min_severity:
                    viol = ",".join(ev.epi_violations) or "-"
                    print(f"\n[EVENT] f{frame_idx} id#{ev.person_id} sev={ev.severity} "
                          f"posture={ev.posture} reba={ev.reba_score}(AL{ev.action_level}) "
                          f"zone={ev.zone_status}/{ev.zone_name or '-'} epi={viol} :: {ev.reason}")
            cv2.putText(annotated, f"{dt_ms:.0f}ms  {fps:.1f}FPS  stride={pipeline._epi_stride}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow(window, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    wall = time.perf_counter() - t_start
    lat = np.array(all_lat) if all_lat else np.array([0.0])
    p95 = float(np.percentile(lat, 95))
    print(f"\n\n{'='*60}")
    print(f"  Frames         : {frame_idx}")
    print(f"  Wall clock     : {wall:.1f}s  ({frame_idx/wall if wall>0 else 0:.1f} FPS effective)")
    print(f"  E2E latency    : mean={lat.mean():.1f}ms  p50={np.percentile(lat,50):.1f}ms  "
          f"p95={p95:.1f}ms  max={lat.max():.1f}ms")
    print(f"  Latency target : {'PASS' if p95 <= config.max_latency_ms else 'FAIL'}  (p95 <= {config.max_latency_ms:.0f}ms)")
    if args.output:
        print(f"  Saved          : {args.output}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
```

---

## Parte 2a — Protocolo de gravação dos 3 cenários

**Regra geral de framing** (vale para os 3): corpo inteiro no quadro, **tornozelos sempre visíveis** (zonas dependem deles), iluminação frontal sem contraluz. Grave a **15 fps** se possível (a histerese e `sustained_seconds` do classificador assumem `target_fps=15`; gravar a 30 fps faz a entrada de 5 frames acontecer em 0,17 s em vez de 0,33 s — não quebra, mas distorce o tempo sustentado relatado). Clipes de **20–30 s** cada.

Thresholds de referência (de `config/pose_config.yaml` → `ClassifierThresholds`):
`trunk_flexion_deg=60`, `spine_load_knee_deg=150`, `deep_squat_knee_deg=90`, `elevated_arm_deg=90`, `trunk_rotation_max=0.35`, `fallen_aspect_ratio=1.5`, `reba_risk_score=8`, `fall_risk_min=0.7`. Histerese: entrada normal **5 frames** (~0,33 s), entrada risco **2 frames** (~0,13 s), saída **15 frames** (~1 s).

### Cenário 1 — Postura ergonomicamente inadequada

A regra **mais fácil e robusta de acionar** é **agachamento prolongado** (`prolonged_squat`): basta `min(joelho) < 90°`, uma única condição. A regra que melhor *conta a história NR-17* é **carga com coluna fletida** (`trunk_flexion_spine_load`), mas ela exige **duas** condições simultâneas: `tronco > 60°` **E** `max(joelho) > 150°` (pernas retas). Grave **as duas** posturas no mesmo clipe.

- **⚠️ Ângulo de câmera (crítico):** a flexão de tronco é medida pelo deslocamento **horizontal** do vetor quadril→ombro contra a vertical da imagem. **Uma inclinação para frente só aparece em vista LATERAL (perfil).** De frente, curvar-se para frente mantém ombro acima do quadril → tronco ≈ 0° e a regra **não dispara**. **Filme o Cenário 1 de lado (90° em perfil).** O agachamento também lê melhor de perfil (joelho hip→knee→ankle).
- **Posturas a executar:** (1) agachar profundamente e segurar ~2 s; (2) em pé, dobrar o tronco para frente >60° com pernas retas (simular pegar caixa do chão sem dobrar joelhos) e segurar ~2 s; (3) braço acima do ombro segurando ~2 s (`elevated_arm_work`, dispara `shoulder_elevation < 90°`).
- **Duração p/ histerese:** segurar cada postura ≥ 1 s (≥ 15 frames @15fps) garante a entrada (5 frames) com folga.
- **EPI:** use capacete/colete para validar que aparecem no `PersonEvent` (depende de o modelo ter classes correspondentes — ver Parte 3 #5).
- **Comando:**
  ```bash
  python src/run_pipeline.py --source data/validation/cenario1.mp4 \
    --output outputs/captures/cenario1_annot.mp4
  ```
- **Esperado na tela / terminal:** label `ergonomically_inadequate`, `reba` 3–8, severidade **2** (laranja); `reason` = `prolonged_squat` / `trunk_flexion_spine_load` / `elevated_arm_work`.

### Cenário 2 — Risco imediato (queda / pessoa caída)

- **Regras exatas:** (1) `person_fallen` → `bbox_width/bbox_height > 1.5` (pessoa horizontal); (2) `fall_velocity` → `fall_risk_score ≥ 0.7` (velocidade vertical + aceleração do centro de massa); (3) `balance_loss_risk` → `trunk_rotation_proxy > 0.35` (torção/assimetria súbita). Entrada com **apenas 2 frames** (recall priorizado).
- **⚠️ Câmera:** quedas **em direção à câmera não alargam a bbox** → `person_fallen` não dispara. **Filme de lado (perfil) ou de um ângulo elevado**, de modo que a pessoa deitada fique claramente mais larga que alta.
- **Simulação segura:** colchão/tatame no chão; queda lateral controlada **sobre o colchão**. Alternativa de menor risco: deitar-se rápido (de lado) e permanecer no chão ~2 s.
- **Duração:** o evento de queda + **permanecer caído ≥ 1,5 s** (a saída exige 15 frames ~1 s; ficar caído confirma o estado e evita flicker).
- **EPI:** capacete recomendado (realismo); não é o foco do cenário.
- **Comando:**
  ```bash
  python src/run_pipeline.py --source data/validation/cenario2.mp4 \
    --output outputs/captures/cenario2_annot.mp4 --min-severity 3
  ```
- **Esperado:** label `immediate_risk`, severidade **4** (vermelho escuro); `reason` = `person_fallen` (ou `fall_velocity` / `balance_loss_risk`). **Esta é a métrica que a mentoria mais vai cobrar (recall ≥ 0,90)** — grave 2–3 quedas no mesmo clipe para ter amostra.

### Cenário 3 — Aproximação a zona crítica

- **Definir a zona ANTES de gravar**, sobre um frame da **mesma câmera/posição/resolução** do clipe:
  ```bash
  python scripts/annotate_zones.py --source data/validation/cenario3.mp4 \
    --output config/zones/zones.json
  # clique os vértices → ENTER → nome/tipo/severidade → q
  ```
  Ou copie o exemplo e ajuste os vértices à sua resolução: `cp config/zones/zones.example.json config/zones/zones.json`.
- **`APPROACH_BUFFER_PX = 80`**: a faixa "aproximando" tem **80 px** a partir da borda do polígono. Em metros depende da escala da cena — se a pessoa ocupa ~400 px de altura (~1,7 m), 80 px ≈ **~0,34 m**. Se a faixa ficar curta demais para demonstrar, aumente `zones.approach_buffer_px` no `pose_config.yaml`.
- **3 sub-transições (fora → aproximando → dentro):** o operador caminha de longe (status `outside`, omitido), entra na faixa de 80 px e **pausa ~1 s** (`approaching`, sev 1), depois pisa **dentro** do polígono com os tornozelos (`inside`, sev ≥ 3).
- **EPI:** a taxonomia §4.2 (faltar `required_ppe` da zona eleva a severidade em +1) **agora está implementada** em `engine.fuse` (ver Parte 3 #5b). Se a zona exige p. ex. `helmet` e a pessoa tem a violação `no_helmet` associada, a severidade sobe +1 (teto 4). Para exercitar isso, defina `required_ppe` na zona e grave o operador **sem** o EPI correspondente — **depende de o modelo emitir as classes `no_*`** (Parte 3 #5).
- **Comando:**
  ```bash
  python src/run_pipeline.py --source data/validation/cenario3.mp4 \
    --zones config/zones/zones.json \
    --output outputs/captures/cenario3_annot.mp4
  ```
- **Esperado:** label muda para `critical_zone` ao entrar; `zone=inside/<nome>`, severidade **3** (vermelho); na faixa, `zone=approaching` severidade 1.

---

## Parte 2b — Protocolo de medição de latência

**E2E (captura → classificação)** já é medido pelo `run_pipeline.py`: o relógio envolve todo `process_frame` (pose + EPI + ângulos + REBA + zona + temporal + classificador + fusão + overlay) e o resumo final imprime mean/p50/p95/max. O `self.last_timings` dá o **breakdown por etapa** (pose / epi / analysis / overlay) ao vivo no HUD. `benchmark_pose.py` continua útil para medir o **modelo de pose isolado** (sem o resto), permitindo isolar quanto do orçamento é o modelo.

**Medição de 60 s com 1, 2 e 3 pessoas** (headless para não pagar custo de render de janela):
```bash
# 60 s @ 15 fps ≈ 900 frames; use uma webcam/clip com N pessoas no quadro
python src/run_pipeline.py --source data/validation/lat_1pessoa.mp4 --no-display --max-frames 900
python src/run_pipeline.py --source data/validation/lat_2pessoas.mp4 --no-display --max-frames 900
python src/run_pipeline.py --source data/validation/lat_3pessoas.mp4 --no-display --max-frames 900

# Pose isolada (baseline do modelo), para comparar:
python scripts/benchmark_pose.py --source data/validation/lat_1pessoa.mp4 --frames 900 --track
```

**Variações a testar** (editar `config/pose_config.yaml`, sem tocar código):
- **CPU vs GPU:** `pipeline.device: cpu` → `cuda` (ou `mps` em Mac). Reexecutar.
- **Modelo s vs n:** `pipeline.pose_model_path: yolo11s-pose.pt` → `yolo11n-pose.pt` (mais rápido, menos preciso). Reexecutar.

**Tabela de resultados a preencher:**

| Config (device / modelo) | Nº pessoas | mean E2E | p95 E2E | pose ms | epi ms | FPS efetivo | ≤200 ms? |
|---|---|---|---|---|---|---|---|
| cpu / yolo11s-pose | 1 | | | | | | |
| cpu / yolo11s-pose | 2 | | | | | | |
| cpu / yolo11s-pose | 3 | | | | | | |
| cpu / yolo11n-pose | 1 | | | | | | |
| cuda / yolo11s-pose | 1 | | | | | | |
| cuda / yolo11s-pose | 3 | | | | | | |

> **Hipótese a confirmar:** em CPU com 1 modelo de pose `s` + EPI, o p95 provavelmente **estoura 200 ms** com 2–3 pessoas; o *stride* adaptativo de EPI mitiga mas não garante. A tabela dirá se a meta exige GPU ou `yolo11n-pose`.

---

## Parte 2c — Ground-truth para `evaluate_pose.py`

### Formato exato que o script espera

`scripts/evaluate_pose.py` lê um **JSON** = lista de anotações **por frame**:
```json
[
  {
    "frame_id": 0,
    "person_id": 1,
    "keypoints": [[x, y], ...],   // 17 pares COCO; [0,0] = não rotulado
    "bbox": [x1, y1, x2, y2],
    "posture": "immediate_risk"   // normal | ergonomically_inadequate | immediate_risk | critical_zone
  }
]
```
Ele indexa por `frame_id`, casa cada anotação a uma pose predita por **IoU de bbox ≥ 0,3** (os `person_id` de GT e do tracker são numerações independentes), e então: calcula **PCK@0.1** (precisa de `keypoints`), **accuracy/recall/F1 por categoria** (precisa de `bbox` + `posture`) e **falsos alarmes/hora** (sobre frames cujo GT é `normal`).

### Mínimo viável para o Dia 1 (recall de risco imediato)

Para **recall** é preciso, por frame: `bbox` (para casar a pose) + `posture`. **`keypoints` só servem ao PCK.** Como anotar bbox+keypoints à mão por frame é inviável no Dia 1, o `scripts/annotate_gt.py` (criado) faz isto em 2 fases:
1. **Fase A (interativa, sem modelo):** você reproduz o clipe e marca, por tecla, a categoria vigente. Rápido.
2. **Fase B (com modelo):** reroda o YOLO-pose sobre o clipe e grava, por frame, `bbox`+`keypoints` da pessoa dominante **com a sua label**.

> **⚠️ Caveat de PCK (honesto):** como os `keypoints` do GT na Fase B vêm do **mesmo detector** que o `evaluate_pose.py` roda, o **PCK fica auto-referencial (~1,0) e NÃO é uma medida válida** de acurácia de keypoints. Este fluxo serve às **métricas de classificação** (recall/accuracy/falso-alarme), onde a sua marcação por tecla é o ground-truth independente. PCK real exige keypoints rotulados à mão (ex.: CVAT) — fica para o Dia 2+ com um subconjunto pequeno (~30–50 frames). Para a mentoria, o número que importa (recall ≥ 0,90) **é** mensurável hoje.

### `scripts/annotate_gt.py` (resumo — arquivo completo no repo)

Teclas: `n/e/i/z` definem a label vigente; `SPACE` pausa; `.` passo; `+/-` velocidade; `u` desfaz; `q` finaliza e salva.
```bash
python scripts/annotate_gt.py \
  --video data/validation/cenario2.mp4 \
  --output data/annotations/cenario2_gt.json \
  --pose-model yolo11s-pose.pt --device cpu
```

### Rodar a avaliação

```bash
python scripts/evaluate_pose.py \
  --annotations data/annotations/cenario2_gt.json \
  --video data/validation/cenario2.mp4 \
  --pose-model yolo11s-pose.pt --device cpu
```

> **Cenário 3:** `evaluate_pose.py` agora aceita `--zones` (default `config/zones/zones.json`). Para avaliar o Cenário 3, aponte para o JSON da zona:
> ```bash
> python scripts/evaluate_pose.py \
>   --annotations data/annotations/cenario3_gt.json \
>   --video data/validation/cenario3.mp4 \
>   --zones config/zones/zones.json --device cpu
> ```

---

## Parte 3 — Correções necessárias (feitas / a observar)

### #1 — `run_video` descarta os `events` → ✅ resolvido (sem editar `run_video`)
`UnifiedPipeline.run_video` ignora o retorno de `process_frame`. Em vez de alterá-lo, `src/run_pipeline.py` conduz o loop e **imprime cada evento** com `severity ≥ --min-severity` (sink de terminal). `run_video` permanece intacto para não quebrar `evaluate_pose.py`.

### #2 — `process_frame` não instrumentava latência por etapa → ✅ resolvido (editado)
Adicionado `self.last_timings` em `src/fusion/pipeline.py`. **Diff aplicado:**
```diff
  self.last_poses: list[PersonPose] = []  # poses from the most recent frame
+ self.last_timings: dict[str, float] = {}  # per-stage latency of the last frame (ms)
```
```diff
+ t_pose = time.perf_counter()
  poses = self._pose.estimate(frame, track=True)
  self.last_poses = poses
+ pose_ms = (time.perf_counter() - t_pose) * 1000

- if self._frame_idx % self._epi_stride == 0:
+ epi_ran = self._frame_idx % self._epi_stride == 0
+ epi_ms = 0.0
+ if epi_ran:
+     t_epi = time.perf_counter()
      ...
+     epi_ms = (time.perf_counter() - t_epi) * 1000

+ t_analysis = time.perf_counter()
  person_bboxes = {p.person_id: p.bbox for p in poses}
  ...
      events.append(event)
+ analysis_ms = (time.perf_counter() - t_analysis) * 1000

+ t_overlay = time.perf_counter()
  annotated = draw_pose_overlay(frame, poses, events, self._zones.get_all())
+ overlay_ms = (time.perf_counter() - t_overlay) * 1000

  elapsed_ms = (time.perf_counter() - t0) * 1000
+ self.last_timings = {"pose_ms": pose_ms, "epi_ms": epi_ms, "epi_ran": epi_ran,
+                      "analysis_ms": analysis_ms, "overlay_ms": overlay_ms, "total_ms": elapsed_ms}
```
A lógica de *stride* adaptativo continua usando `elapsed_ms` (total). Não-quebra para `evaluate_pose.py` (assinatura de `process_frame` inalterada; o campo extra é só leitura opcional).

### #3 — `ZoneManager` quebra sem `zones.json`? → ✅ não quebra
`ZoneManager.__init__` só carrega se o arquivo existir; senão fica com lista vazia, `get_all()` retorna `[]`, `ZoneDetector` não retorna nada e `draw_pose_overlay` trata zonas vazias. **Nenhuma correção necessária.** Para o Cenário 3, criado `config/zones/zones.example.json` (zona retangular 1280×720, severidade 4) — copiar para `zones.json` e ajustar vértices, ou usar `annotate_zones.py`.

### #4 — `yolo11s-pose.pt` baixa em runtime → ✅ mitigado no Passo 0
Baixa na 1ª chamada `YOLO(...)` para o **CWD**. Pré-baixado no Passo 0 para não dar timeout durante a gravação. **Sempre rodar a partir da raiz do projeto** (caminhos relativos).

### #5 — Discrepância de classes do modelo EPI → ⚠️ **VERIFICAR no início (bloqueia o cruzamento EPI+pose)**
Você descreve `exp.pt` com 7 classes **sem** rótulos `no_*`; `DecisionEngine.EPI_VIOLATION_LABELS = {no_gloves, no_goggle, no_helmet, no_boots, none}`. Se o modelo **não** emite classes `no_*`, então `event.epi_violations` fica **sempre vazio** e a severidade por EPI (sev 2) **nunca dispara** — o cruzamento EPI+pose dos cenários 1 e 3 não aparece. **Não consegui introspectar (deps ausentes).**
- **Ação:** rodar o one-liner do Passo 0 (`print(YOLO('models/exp.pt').names)`).
- **Se confirmar ausência de `no_*`:** para a demo do Dia 1, tratar o EPI como detecção de itens presentes (capacete/colete aparecem nas bboxes do EPI, mas sem lógica de violação), e registrar como item a corrigir (alinhar `EPI_VIOLATION_LABELS`/`construction-ppe.yaml` ao modelo real, ou usar um modelo com classes de violação). **Não é bloqueador da gravação de pose/queda/zona.**

### #5b — Elevação de severidade por `required_ppe` → ✅ implementada (editado)
A taxonomia §4.2 promete "+1 de severidade se falta o `required_ppe` da zona". Implementado:
- `ZoneResult` ganhou o campo `required_ppe`, populado pelo `ZoneDetector` a partir de `zone.required_ppe`.
- `engine.fuse`: estando dentro/perto de uma zona (`worst`), se algum item exigido tem a violação correspondente em `event.epi_violations` (mapeamento `helmet` → `no_helmet`), `sev = min(sev + 1, 4)`.

**Diff (resumo):**
```diff
# src/zones/detector.py
- from dataclasses import dataclass
+ from dataclasses import dataclass, field
  class ZoneResult:
      ...
+     required_ppe: list[str] = field(default_factory=list)
  # ambos os ZoneResult em check() agora passam required_ppe=zone.required_ppe

# src/fusion/engine.py
- if zone_results:
+ worst = None
+ if zone_results:
      worst = max(zone_results, key=...)
  ...
+ if worst is not None and worst.required_ppe and event.epi_violations:
+     missing = [ppe for ppe in worst.required_ppe if f"no_{ppe}" in event.epi_violations]
+     if missing:
+         sev = min(sev + 1, 4)
  event.severity = sev
```
> Depende do mapeamento de nomes: a zona deve usar stems que casem com as violações (`helmet`/`gloves`/`goggle`/`boots` → `no_helmet`/`no_gloves`/`no_goggle`/`no_boots`). O `zones.example.json` já usa `["helmet","boots"]`. **Só dispara se o modelo EPI emitir classes `no_*`** (Parte 3 #5).

### #5c — `evaluate_pose.py` sem `--zones` → ✅ implementada (editado)
Adicionada a flag `--zones` (default `config/zones/zones.json`), repassada a `PipelineConfig(zones_config=...)`. Necessária para pontuar o cenário `critical_zone`.

---

## Ordem de execução do dia (~6 h)

| Hora | Bloco | Tarefa | Saída / gate |
|---|---|---|---|
| **0:00–0:30** | Setup | Passo 0: venv, `pip install`, pré-baixar `yolo11s/n-pose`, **verificar `exp.pt`.names** e GPU. | Imports OK; classes do EPI conhecidas; cuda? sim/não. |
| **0:30–1:00** | Smoke test | `run_pipeline --source 0` (webcam, com janela). Confirmar esqueleto, REBA, HUD de latência, eventos no terminal. | CLI funciona ponta a ponta. **Gate:** se falhar, parar e corrigir antes de gravar. |
| **1:00–1:15** | Zona | `annotate_zones.py` na posição de câmera do Cenário 3; salvar `config/zones/zones.json`. | 1 zona persistida. |
| **1:15–2:30** | Gravação | **Cenário 1** (lateral: agachar / tronco fletido pernas retas / braço elevado, segurando ≥1 s cada) + rodar `run_pipeline --output`. Conferir labels/severidade no vídeo anotado. | `cenario1.mp4` + `cenario1_annot.mp4` validados. |
| **2:30–3:30** | Gravação | **Cenário 2** (queda lateral sobre colchão, 2–3 repetições, permanecer caído ≥1,5 s) + `run_pipeline --min-severity 3`. **Métrica crítica.** | `cenario2.mp4` + anotado; `immediate_risk` confirmado visualmente. |
| **3:30–4:15** | Gravação | **Cenário 3** (fora → aproximando ~1 s → dentro) + `run_pipeline --zones`. | `cenario3.mp4` + anotado; transições visíveis. |
| **4:15–5:15** | Latência | Medir E2E headless com 1/2/3 pessoas; variar cpu/cuda e s/n; preencher tabela 2b. | **Tabela de latência preenchida** (bloqueador Dia 2). |
| **5:15–6:00** | Ground-truth | `annotate_gt.py` no Cenário 2 (foco recall) + `evaluate_pose.py`; se sobrar tempo, cenários 1 e 3. | `cenario2_gt.json` + 1ª leitura de recall. |

**Gates de fim de dia (bloqueadores do Dia 2):**
1. ✅ 3 vídeos brutos + 3 anotados gravados e visualmente validados.
2. ✅ Tabela de latência E2E preenchida (sabe-se se a meta de 200 ms é atingível e em qual config).
3. 🎯 Bônus: 1ª medição de recall de `immediate_risk` via `evaluate_pose.py`.

**Se o tempo apertar, priorize nesta ordem:** gravar os 3 vídeos > medir latência > ground-truth/recall. Os vídeos não são refazíveis depois (equipe presente); latência e recall podem ser recalculados sobre os vídeos no Dia 2.
