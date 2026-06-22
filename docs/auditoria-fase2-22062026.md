# Auditoria Técnica Fase 2 — Cypher-Capture

**Data:** 22/06/2026
**Auditado por:** Claude Code (Opus 4.8)
**Versão repositório:** `196a5f9` (branch `main`, working tree limpo)
**Escopo:** revisão code-level de todo o `src/`, `scripts/`, `config/` e `docs/`

---

## 0. Achado central (ler primeiro)

> **A premissa do prompt de auditoria está desatualizada.** A Fase 2 **não está por implementar** — ela já foi implementada e *merged* na `main` via PR #1 (`claude/pose-estimation-behavioral-analysis`). O repositório contém pose estimation, motor REBA completo, classificador de posturas em 4 categorias com histerese, análise temporal, detecção de zonas críticas, motor de fusão e um pipeline unificado EPI+pose, além de scripts de avaliação e benchmark.

O que **falta** não é código de Fase 2, e sim **evidência de que ele funciona**: não há testes, não há dados de validação anotados e, portanto, **nenhuma das métricas-alvo (PCK ≥ 0,80; Recall risco ≥ 0,90; Latência ≤ 200 ms) foi efetivamente medida**. Os números são, hoje, aspiracionais. Esse é o risco dominante da fase.

Reformulação prática da pergunta "iniciar Fase 2?": a fase a iniciar não é *construção*, é **validação, instrumentação de produção (persistência/API) e endurecimento de privacidade (LGPD)**.

---

## 1. Estrutura de Diretórios

```
cypher-capture/
├── config/
│   ├── construction-ppe.yaml      # Dataset YAML Fase 1 (11 classes, paths)
│   ├── pose_config.yaml           # ★ Fonte única de thresholds Fase 2
│   └── zones/                     # zones.json (vazio — só .gitkeep)
├── data/
│   ├── annotations/               # .gitkeep — VAZIO (sem ground-truth)
│   └── validation/                # .gitkeep — VAZIO (sem vídeos de validação)
├── docs/
│   └── taxonomy.md                # ★ Taxonomia de posturas + REBA (PT-BR)
├── models/
│   └── exp.pt                     # ★ Modelo EPI treinado (~52 MB, commitado)
├── notebooks/                     # .gitkeep — vazio
├── outputs/{captures,runs}/       # .gitkeep — runtime
├── scripts/
│   ├── annotate_zones.py          # Ferramenta interativa de desenho de zonas
│   ├── benchmark_pose.py          # Benchmark de latência do modelo pose
│   ├── download_dataset.py        # Download dataset Construction-PPE
│   ├── evaluate_pose.py           # ★ Harness de métricas (PCK/recall/latência)
│   └── run.sh                     # Wrapper de inferência Fase 1
└── src/
    ├── detect.py                  # ★ Entry-point Fase 1 (EPI standalone)
    ├── train.py                   # Treino YOLO
    ├── fusion/
    │   ├── engine.py              # DecisionEngine + PersonEvent (schema de saída)
    │   └── pipeline.py            # ★ UnifiedPipeline (loop EPI+pose unificado)
    ├── pose/
    │   ├── estimator.py           # PoseEstimator (YOLO11-pose + ByteTrack)
    │   ├── angles.py              # Extração de ângulos articulares
    │   ├── reba.py                # Motor REBA (tabelas A/B/C)
    │   ├── classifier.py          # Classificador de postura + histerese
    │   └── temporal.py            # Análise temporal (queda, repetitividade)
    ├── zones/
    │   ├── manager.py             # Persistência JSON de zonas
    │   └── detector.py            # Teste ponto-em-polígono (INSIDE/APPROACHING)
    └── utils/
        ├── video.py               # open_capture / resolve_source
        ├── visualization.py       # Desenho de bbox EPI (Fase 1)
        ├── pose_visualization.py  # Overlay de esqueleto + severidade (Fase 2)
        └── upload.py              # Upload Cloudinary
```

**Observações de organização**
- Separação por domínio limpa e idiomática (`pose/`, `zones/`, `fusion/`, `utils/`), com `__init__.py` exportando a API pública de cada pacote. Boa base para crescer.
- **Dois caminhos de execução paralelos e desconexos:** `src/detect.py` (Fase 1, EPI puro, com upload Cloudinary) e `src/fusion/pipeline.py::UnifiedPipeline` (Fase 2, EPI+pose). Eles não se conhecem — ver §4.
- `models/exp.pt` é o **único** peso commitado e é o detector de **EPI** (não o de pose). O peso de pose (`yolo11s-pose.pt`) **não está no repo**: é baixado automaticamente pela Ultralytics no primeiro `YOLO(...)` (requer rede na primeira execução).
- `config/zones/zones.json` ainda não existe (só `.gitkeep`); `ZoneManager` lida com a ausência sem quebrar (carrega lista vazia).

---

## 2. Inventário Fase 1 (Detecção de EPIs)

| Arquivo | LoC | Dependências | Descrição funcional |
|---|---|---|---|
| [src/detect.py](../src/detect.py) | 181 | `ultralytics`, `cv2`, `requests`, `dotenv` | Entry-point CLI da Fase 1. Abre fonte (usb/vídeo/imagem/dir), roda YOLO frame a frame, desenha detecções, detecta violação e faz upload com *cooldown*. |
| [src/utils/visualization.py](../src/utils/visualization.py) | 45 | `cv2`, `ultralytics.Results` | `draw_detections` + `class_color`. Define `VIOLATION_CLASSES`. Verde=compliant, vermelho=violação, âmbar=Person. |
| [src/utils/upload.py](../src/utils/upload.py) | 67 | `cloudinary`, `cv2`, `dotenv` | `DetectionSnapshot` (par raw+annotated), `build_snapshot`, `upload_snapshot`. UID = timestamp + md5 do frame raw. |
| [src/train.py](../src/train.py) | 63 | `ultralytics` | Fine-tuning YOLO sobre `construction-ppe.yaml`. |
| [scripts/download_dataset.py](../scripts/download_dataset.py) | 44 | `urllib`, `zipfile` | Baixa/extrai dataset (~178 MB). |
| [scripts/run.sh](../scripts/run.sh) | — | bash | Wrapper de `detect.py` via env vars. |

**Notas de código sobre a Fase 1:**
- **`detect.py` NÃO usa ByteTrack.** Chama `model(frame)`, não `model.track(...)`. O parâmetro `track_id` em `handle_upload` é, na verdade, o `frame_id`, e o `upload_cooldown` é chaveado por uma constante `0` ([detect.py:117](../src/detect.py#L117)) — ou seja, o *cooldown* é global por stream, não por pessoa. Rastreamento multi-pessoa só existe no caminho de **pose** (§3).
- O upload envia o **frame raw completo** (rostos legíveis) para o Cloudinary — relevante para LGPD (§9).
- A Fase 1 é funcional como demo de detecção; o modelo `exp.pt` está presente. mAP@0.5 = 0.809 é citado no contexto do projeto (não há artefato de métricas commitado no repo para confirmar).

---

## 3. Presença de Código Fase 2 (checklist comentado)

| Item | Status | Onde / observação |
|---|---|---|
| Modelo YOLO11-pose carregado/referenciado | ✅ **Funcional** | [estimator.py:58](../src/pose/estimator.py#L58) `YOLO(model_path)`; default `yolo11s-pose.pt`. Peso não commitado (auto-download). |
| Cálculo de ângulos articulares | ✅ **Funcional** | [angles.py](../src/pose/angles.py): tronco (vs. vertical), pescoço, ombro, cotovelo, joelho (bilateral), proxy de rotação. Política `None` para keypoint < 0,5 conf. |
| Implementação REBA | ✅ **Funcional** | [reba.py](../src/pose/reba.py): tabelas A/B/C completas, score 1–12, 4 níveis de ação. Limitações 2D declaradas (punho fixo=1, sem carga). RULA **não** implementado (só REBA). |
| Classificador de categorias de postura | ✅ **Funcional** | [classifier.py](../src/pose/classifier.py): `NORMAL / ERGONOMICALLY_INADEQUATE / IMMEDIATE_RISK / CRITICAL_ZONE`, máquina de estados por pessoa com histerese assimétrica (entrada rápida / saída lenta). |
| Análise temporal | ✅ **Funcional** | [temporal.py](../src/pose/temporal.py): janela deslizante 60 frames, velocidade/aceleração do CoM, fall-risk heurístico, repetitividade por autocorrelação. |
| Motor de zonas críticas | ✅ **Funcional** | [zones/](../src/zones/): `ZoneManager` (persistência JSON) + `ZoneDetector` (`pointPolygonTest`, INSIDE/APPROACHING com buffer). Polígonos cacheados. |
| Rastreamento multi-pessoa (ByteTrack) | ✅ **Funcional (só no pose)** | [estimator.py:65](../src/pose/estimator.py#L65) `model.track(..., tracker="bytetrack.yaml", persist=True)`. IDs de fallback negativos evitam colisão com IDs reais. |
| Pipeline unificado (EPI + pose no mesmo stream) | ⚠️ **Parcial** | [pipeline.py](../src/fusion/pipeline.py): `UnifiedPipeline` roda pose todo frame + EPI com *stride* adaptativo. **Mas: sem entry-point CLI e sem persistência de eventos** (§4). |
| Motor de fusão de decisão | ✅ **Funcional** | [engine.py](../src/fusion/engine.py): associa EPI↔pessoa por *containment*, agrega severidade 0–4. |
| Ferramenta de anotação de zonas | ✅ **Funcional** | [annotate_zones.py](../scripts/annotate_zones.py): desenho interativo, preserva zonas existentes. |
| Harness de avaliação | ✅ **Presente, nunca executável hoje** | [evaluate_pose.py](../scripts/evaluate_pose.py): PCK@0.1, accuracy, recall por categoria, false-alarm por episódio, latência mean/p95. **Requer GT + vídeo que não existem no repo.** |
| Visualização de pose | ✅ **Funcional** | [pose_visualization.py](../src/utils/pose_visualization.py): esqueleto COCO, cor por severidade, label REBA/postura/zona. |
| RULA | ❌ **Ausente** | Apenas REBA. Taxonomia não exige RULA; ok se REBA for suficiente para NR-17. |
| Persistência (Redis/DB) | ❌ **Ausente** | `PersonEvent` tem schema "targets Phase 3 persistence" mas nada o consome (§4). |
| API (FastAPI) | ❌ **Ausente** | Planejada; não há código. Fase 1 faz POST opcional a `API_ENDPOINT`. |

---

## 4. Análise do Pipeline de Inferência

Há **dois** loops. O relevante para a Fase 2 é `UnifiedPipeline.process_frame` ([pipeline.py:101](../src/fusion/pipeline.py#L101)).

### Fluxo de `process_frame`

```
                       ┌─────────────────────┐
  frame ──────────────►│  PoseEstimator      │  (model.track, TODO frame)
                       │  .estimate(track)   │──► list[PersonPose] + ByteTrack IDs
                       └─────────┬───────────┘
                                 │
            frame_idx % stride==0│  (EPI roda a cada N frames; N adaptativo)
                       ┌─────────▼───────────┐
                       │  EPI YOLO (exp.pt)  │──► bboxes + labels (cache p/ frames pulados)
                       └─────────┬───────────┘
                                 │
                   associate_epis (containment EPI↔pessoa)
                                 │
              ┌──────────────────┼─────────── por pessoa ──────────────┐
              ▼                  ▼                                      ▼
        extract_angles ──► compute_reba          ZoneDetector.check(feet)
              │                  │                       │
              ▼                  ▼                       ▼
        TemporalAnalyzer ──► fall_risk        PostureClassifier.update(
        (usa estado do frame ANTERIOR)          angles, reba, bbox, in_zone, fall_risk)
                                 │                       │
                                 └────────► DecisionEngine.fuse ──► PersonEvent (severidade 0–4)
                                                         │
                       draw_pose_overlay ◄───────────────┘
                                 │
                          (annotated, events)
```

### Respostas às perguntas do prompt

- **Estrutura do loop:** captura → pose (todo frame) → EPI (a cada N) → ângulos → REBA → zona → temporal → classificador → fusão → overlay. Limpo e legível.
- **Quantos modelos / ordem:** **2 modelos** — pose (`yolo11s-pose.pt`) carregado no `__init__`; EPI (`exp.pt`) carregado *lazy* no 1º frame (`_ensure_epi_model`). Pose roda **primeiro e sempre**; EPI roda com *stride* ≥ 1.
- **Separação de etapas:** aquisição (`run_video`) e processamento (`process_frame`) separados. **Mas saída e persistência não:** `run_video` **descarta** os `events` retornados ([pipeline.py:180](../src/fusion/pipeline.py#L180)). Não há sink (DB/Redis/API/arquivo). Os eventos só existem em memória durante o frame.
- **Latência por etapa:** **não instrumentada por etapa.** Só o total do frame é medido, e apenas para alimentar o controle adaptativo de *stride* ([pipeline.py:163](../src/fusion/pipeline.py#L163)). `benchmark_pose.py` mede só o modelo de pose isolado. Não há breakdown pose vs. EPI vs. classificação.
- **Thread-safe?** **Não.** `model.track(persist=True)` mantém estado interno do tracker; `_states`, `_pending`, buffers temporais são dicts mutáveis compartilhados sem lock. Projetado para **um stream sequencial**. Processar frames em paralelo corromperia o estado por pessoa.
- **Redis/DB?** **Nenhum.** Tudo em memória. `PersonEvent` foi desenhado com timestamp/campos prontos para serialização ("targets Phase 3 persistence") mas não há consumidor.

### Lacunas / gargalos do pipeline
1. **Sem entry-point CLI para o pipeline unificado.** `UnifiedPipeline.run_video` existe, mas nada na linha de comando o invoca — só `evaluate_pose.py` e construção manual. Falta um `python src/run_pipeline.py --source ...`.
2. **Eventos não persistidos** (sink ausente) — bloqueia qualquer integração com Fase 3.
3. **Controle de latência reativo, não preditivo:** se o frame já estourou 200 ms, o *stride* só aumenta no **próximo** frame; o frame atual já passou do orçamento.
4. **CPU por default** (`device: cpu`). yolo11s-pose + EPI em CPU dificilmente fica ≤ 200 ms p95 com múltiplas pessoas — risco direto à meta de latência (§10-E).
5. **Redundância de detecção de pessoa:** o modelo EPI (classe `Person`) e o de pose ambos detectam pessoas; as caixas `Person` do EPI são associadas e descartadas.

---

## 5. Dependências e Imports

Conteúdo de [requirements.txt](../requirements.txt):

```
ultralytics>=8.3.0
opencv-python>=4.8.0
python-dotenv>=1.0.0
cloudinary>=1.44.0
requests>=2.32.0
numpy>=1.24.0
Pillow>=10.0.0
scipy>=1.10.0
PyYAML>=6.0
```

**Observações**
- **Versões com piso (`>=`), não fixadas.** Para reprodutibilidade de métricas (e auditoria com mentores), recomenda-se *pinning* exato (`==`) ou um *lock* — sobretudo `ultralytics` e a versão implícita de `torch` que ela arrasta, que mudam comportamento de modelo entre releases.
- **PyTorch não aparece** explicitamente: vem transitivamente via `ultralytics`. Recomenda-se declará-lo explicitamente com versão + variante (CPU vs. CUDA), já que latência depende disso.
- **Pose estimation:** nenhuma lib dedicada (MediaPipe/MMPose/OpenPose) — a pose vem 100% da Ultralytics (YOLO11-pose). Coerente com a stack e evita dependência pesada.
- **Geometria:** usa `numpy` + `cv2.pointPolygonTest`. **Shapely não é usado** — e, dado que a detecção de zonas é só ponto-em-polígono e distância, **não é necessário**. (O prompt levantava essa hipótese; descartada.)
- **`scipy` está declarado mas não é importado em nenhum arquivo** (`temporal.py` usa `np.correlate`, não `scipy.signal`). Dependência morta hoje — remover ou passar a usar (`scipy.signal` daria autocorrelação/FFT mais robusta para repetitividade).
- **Faltam dependências de teste** (`pytest`, etc.) — coerente com a ausência de testes (§6).

---

## 6. Testes Unitários / Integração

**Status: ❌ Inexistentes.** Não há diretório `tests/`, nem `test_*.py`, nem `conftest.py`, nem fixtures de vídeo/imagem. Coverage = 0%.

Isso é a lacuna mais crítica para uma fase cujas metas são numéricas (PCK/recall/latência). Funções 100% determinísticas e puras — **ideais para teste unitário sem GPU/modelo** — estão completamente sem cobertura:

- `reba.py::compute_reba` — entradas conhecidas → score esperado (validável contra a tabela oficial Hignett & McAtamney).
- `angles.py::_angle`, `extract_angles` — geometria pura; casos sintéticos (braço na horizontal = 90°, etc.).
- `classifier.py` — máquina de estados/histerese: alimentar sequências de frames e assertar transições (entra em 2 frames no risco, sai só após 15).
- `engine.py::_containment`, `associate_epis` — atribuição EPI↔pessoa.
- `zones/detector.py` — ponto dentro/fora/próximo de polígono conhecido.
- `temporal.py::_autocorrelation_score` — sinal senoidal sintético → score alto; ruído → score baixo.

**Recomendação:** criar `tests/` com `pytest` cobrindo prioritariamente REBA, classificador (histerese) e zonas — alta densidade de regras de negócio, custo de teste baixíssimo, e é exatamente onde um bug silencioso destrói a meta de recall.

---

## 7. Documentação Técnica

| Artefato | Avaliação |
|---|---|
| [docs/taxonomy.md](taxonomy.md) | ✅ **Excelente.** 187 linhas em PT-BR: keypoints COCO, definição de cada ângulo, tabelas REBA, as 4 categorias com condição/justificativa/keypoints citando NR-17 §17.6, máquina de estados, e uma tabela honesta de *limitações declaradas* (visão 2D, punho, oclusão). É a peça de documentação mais forte do repo e serve de espec para a Fase 2. |
| [README.md](../README.md) | ⚠️ **Desatualizado.** Descreve **apenas a Fase 1** (PPE). A "Project Structure" no README **não menciona** `pose/`, `zones/`, `fusion/`, `pose_config.yaml` nem os scripts de avaliação. Um dev novo lendo só o README não saberia que a Fase 2 existe. |
| Docstrings | ✅ **Acima da média.** Módulos têm docstring de topo explicando intenção; decisões não óbvias têm comentário com o *porquê* (ex.: por que *containment* e não IoU; por que IDs negativos; por que `abs(dy)`; por que a autocorrelação só conta após o zero-crossing). Qualidade de comentário consistentemente boa. |
| `docs/` diagrams | ❌ Sem diagrama de arquitetura formal (o da taxonomia é ASCII da máquina de estados). |

**Gap principal:** o README precisa de uma seção Fase 2 (como rodar o pipeline unificado, anotar zonas, rodar avaliação). Hoje há um descompasso grande entre o que o código faz e o que a documentação de entrada anuncia.

---

## 8. Configuração e Variáveis de Ambiente

- **[config/pose_config.yaml](../config/pose_config.yaml)** — ✅ **fonte única de thresholds.** `PipelineConfig.from_yaml` ([pipeline.py:47](../src/fusion/pipeline.py#L47)) mapeia 1:1 os campos; chaves desconhecidas são ignoradas. Excelente: todo limiar ergonômico (NR-17), de latência, de histerese e de zona é configurável sem tocar código. Commit `20fc372` consolidou isso.
- **[config/construction-ppe.yaml](../config/construction-ppe.yaml)** — dataset/classes Fase 1.
- **[.env.example](../.env.example)** — só credenciais Cloudinary + `API_ENDPOINT` opcional. **Não cobre Fase 2** (device, caminhos de modelo já estão no YAML, o que é ok).
- **Hardcoded:** defaults sensatos em dataclasses (`PipelineConfig`, `ClassifierThresholds`, `APPROACH_BUFFER_PX=80`) mas todos sobrescrevíveis via YAML. `models/exp.pt` e `config/zones/zones.json` são caminhos default no YAML. Aceitável.
- **Configs por modelo:** EPI vs. pose separados (`epi_model_path`, `pose_model_path`, `conf_epi`, `conf_pose`). ✅

**Parâmetros configuráveis prontos para Fase 2** (de `pose_config.yaml`): `device`, modelos, `conf_*`, `target_fps`, `max_latency_ms`, `epi_stride_max`, `enter_frames`/`enter_frames_risk`/`exit_frames`, e os 8 thresholds ergonômicos + `approach_buffer_px`.

---

## 9. LGPD / Anonimização

| Aspecto | Status | Detalhe |
|---|---|---|
| IDs efêmeros | ✅ Parcial | IDs do ByteTrack não persistem entre sessões; taxonomy.md afirma conformidade. Correto **para o caminho de pose**. |
| Blur de rosto | ❌ Ausente | Nenhum *blur*/pixelização. O overlay desenha keypoints faciais (nose/eyes/ears) sobre o rosto, mas o frame por baixo é o original. |
| Redimensionamento/minimização de imagem sensível | ❌ Ausente | — |
| **Upload de identidade a terceiro** | ⚠️ **Risco** | [upload.py](../src/utils/upload.py) + [detect.py:120](../src/detect.py#L120) enviam o **frame raw completo** (rostos identificáveis) ao **Cloudinary** (serviço externo, fora do controle do projeto). Sob LGPD, isso é tratamento + transferência de dado pessoal sem anonimização nem base legal explícita no código. |
| Logs com identidade | ⚠️ Baixo | Logs imprimem `person_id` numérico (pseudônimo) e URLs de upload — não expõem nome, mas as URLs apontam para imagens com rosto. |
| Comentários sobre LGPD | ✅ | Apenas em `taxonomy.md` (IDs efêmeros). Sem política no código de upload. |

**Gaps a endereçar na Fase 2:**
1. **Blur de rosto antes de qualquer persistência/upload** (usar os keypoints faciais COCO já disponíveis para localizar a região — é "de graça" no pipeline de pose).
2. **Reavaliar o upload de frame raw ao Cloudinary**: anonimizar antes de enviar, ou manter armazenamento on-prem, ou registrar base legal/retenção.
3. O caminho de **pose** ainda **não tem upload** — bom momento para nascer já anonimizado, em vez de herdar o padrão da Fase 1.

---

## 10. Recomendações Estruturais para Fase 2

### A. Módulos Fase 1 reutilizáveis / a estender
- **`utils/video.py`** (`open_capture`) — já é compartilhado pelos dois caminhos. Manter.
- **`utils/upload.py`** — reutilizável, **mas** precisa de *hook* de anonimização antes de servir à Fase 2 (§9).
- **`exp.pt` + classes EPI** — já consumido pelo `UnifiedPipeline` via `associate_epis`. Funciona.
- **Estrutura de evento** — `PersonEvent` já generaliza EPI+pose+zona+REBA; é o ponto de integração natural com a Fase 3.

### B. A criar do zero
- **Camada de persistência / sink de eventos** (Redis ou DB + opcionalmente fila) — hoje os `PersonEvent` são descartados.
- **API (FastAPI)** expondo eventos/stream — planejada, inexistente.
- **Entry-point CLI do pipeline unificado** (`src/run_pipeline.py` ou subcomando) — falta o "main" que a Fase 2 deveria expor.
- **Suíte de testes** (`tests/`) — §6.
- **Dataset de validação anotado** (`data/annotations/*.json` + `data/validation/*.mp4`) — sem isso, `evaluate_pose.py` nunca roda e nenhuma meta é verificável.
- **Módulo de anonimização** (blur facial) — §9.

### C. Refatorações recomendadas antes de avançar
1. **Unificar os dois caminhos de execução.** Hoje `detect.py` (Fase 1) e `UnifiedPipeline` (Fase 2) divergem: o primeiro não rastreia (sem ByteTrack), o segundo sim. Decidir se `detect.py` vira um modo do pipeline unificado ou é aposentado, para não manter duas lógicas de EPI.
2. **Extrair uma abstração de *sink*** (`EventSink` com implementações `Stdout`/`Json`/`Redis`/`Http`) e fazer `run_video` empurrar os eventos para ela — destrava persistência sem reescrever o loop.
3. **Instrumentar latência por etapa** (pose / EPI / pós-processamento) para diagnosticar a meta de 200 ms com dados, não suposição.
4. **Atualizar o README** com a Fase 2 (onboarding).
5. **Pin de dependências** + declarar `torch` explicitamente (reprodutibilidade de métricas).
6. Remover `scipy` morto **ou** usá-lo em `temporal.py`.

### D. Ordem de implementação sugerida
Como o *core* algorítmico já existe, a sequência muda de "construir" para "provar e produtizar":

1. **Validação primeiro (desbloqueia tudo):** montar `data/validation/` + `data/annotations/` e rodar `evaluate_pose.py`. Sem isso, não há como saber se PCK/recall/latência são atingidos — e ajustes posteriores ficam às cegas.
2. **Testes unitários** das regras puras (REBA, histerese, zonas) — barato e protege o recall.
3. **Calibração de thresholds** com base nos resultados de (1): a meta de **recall ≥ 0,90 em risco imediato** manda; ajustar `enter_frames_risk`, `fallen_aspect_ratio`, `fall_risk_min`, `trunk_rotation_max`.
4. **Decisão de latência:** se p95 > 200 ms (provável em CPU), habilitar GPU (`device: cuda`) ou trocar para `yolo11n-pose.pt`; medir de novo.
5. **Sink de eventos + entry-point CLI** (produtização do loop).
6. **Anonimização (LGPD)** antes de qualquer persistência de imagem.
7. **API (FastAPI)** sobre o sink.
8. **Atualização de docs/README.**

> Já que ByteTrack, ângulos e REBA estão prontos e integrados, a prioridade **não** é mais "ângulos → REBA" — é **validar o que existe e endurecer recall/latência/LGPD**.

### E. Riscos técnicos
| Risco | Severidade | Por quê |
|---|---|---|
| **Métricas nunca medidas** (sem GT/validação) | 🔴 Alta | PCK/recall/latência são hoje aspiracionais; impossível afirmar conformidade com as metas da fase. |
| **Latência em CPU > 200 ms** | 🔴 Alta | `device: cpu` default; 2 modelos por stream. O *stride* adaptativo degrada o EPI mas não garante o p95 com várias pessoas. |
| **Recall de queda em câmera frontal** | 🟠 Média | Regra principal de queda é `bbox_w/h > 1.5`; quedas em direção à câmera não alargam a bbox. Mitigação parcial via `fall_risk` temporal (heurística não validada). |
| **Sem testes** | 🟠 Média | Mudança de threshold pode quebrar recall silenciosamente. |
| **Não thread-safe** | 🟡 Baixa-Média | Limita a 1 stream sequencial; escalar a N câmeras exige 1 instância isolada por stream (estado por pessoa não é compartilhável). |
| **LGPD (upload de rosto)** | 🟠 Média | Exposição regulatória, não técnica, mas precisa entrar no escopo da fase. |
| **`yolo11s-pose.pt` baixado em runtime** | 🟡 Baixa | 1ª execução exige rede; ambientes air-gapped quebram. Commitar/versionar o peso. |

---

## 11. Resumo Executivo

| Item | Status | Observação |
|---|---|---|
| **Fase 1 funcional?** | ✅ | Detector EPI (`exp.pt`) + `detect.py` operam; mAP 0.809 citado (não há artefato no repo). Não usa ByteTrack. |
| **Código pose pronto?** | ✅ | Pose, ângulos, REBA, classificador, temporal, zonas e fusão **implementados e merged**. Pouca refatoração de *core* necessária. |
| **Pipeline unificado?** | ⚠️ | `UnifiedPipeline` roda EPI+pose no mesmo loop, mas **sem entry-point CLI e sem persistência** dos eventos (descartados). |
| **Testes?** | ❌ | Zero. Nem `tests/`, nem fixtures, coverage 0%. Maior dívida técnica. |
| **Documentação?** | ⚠️ | `taxonomy.md` excelente; docstrings boas; **README desatualizado** (só Fase 1) — onboarding parcial. |
| **LGPD ready?** | ⚠️ | IDs efêmeros ok; **sem blur facial** e **upload de frame raw a terceiro (Cloudinary)** sem anonimização. |
| **Risk score (0–10)** | **6/10** | Confiança **moderada** em atingir as metas. O algoritmo existe e é bem-feito; o risco é inteiramente *não verificado*: nenhuma métrica medida, latência em CPU duvidosa, recall não calibrado, zero testes. |

### Recomendação

**Iniciar Fase 2 — SIM, com condicionantes**, e com o escopo redefinido. O trabalho de **construção** do *core* de pose/ergonomia **já foi feito**; o que resta é a fase de **engenharia de produto e garantia**:

1. **Condicionante bloqueante:** montar dataset de validação anotado e rodar `evaluate_pose.py` **antes** de qualquer otimização — é o que transforma "achamos que atinge" em "atinge". Sem isso, não há base para o checkpoint com mentores.
2. **Decidir device/modelo de pose** com a latência medida em mãos (provável necessidade de GPU ou `yolo11n-pose`).
3. **Criar suíte de testes** das regras puras (protege recall ≥ 0,90).
4. **Tratar LGPD** (blur + revisão do upload) dentro da fase, não depois.
5. **Produtizar o loop** (sink de eventos + CLI) para destravar a Fase 3.

> **Mensagem para o checkpoint quinzenal (Fernando V. Marcolina / Wendel de Almeida Passos):** o diferencial deste repo é que o *core* algorítmico da Fase 2 já está implementado, documentado contra NR-17/NR-35 e com limitações 2D declaradas honestamente. A conversa de mentoria deve focar em **como validar** (montagem do conjunto de ground-truth, protocolo de medição de PCK/recall/latência) e na **decisão de hardware** para a meta de 200 ms — não em arquitetura de algoritmo, que está madura.

---

*Auditoria realizada por leitura code-level de 100% dos arquivos `.py`, configs e docs do repositório no commit `196a5f9`.*
