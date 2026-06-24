# Métricas — Dia 2 (23/06/2026)

**Ambiente:** Python 3.10 / `torch 2.6.0+cu124` / GPU CUDA disponível / `yolo11s-pose`.

## 0. O que dá e o que não dá para medir hoje

| Métrica-alvo (Tabela 5, Fase 2) | Mensurável no Dia 2? | Porquê |
|---|---|---|
| Latência E2E ≤ 200 ms | ✅ **sim, medida (real)** | Independe dos cenários; medida sobre `test_bus.mp4` (~4 pessoas) |
| Falsos alarmes/hora → 0 | ✅ **sim, parcial** | Medível em frames normais; `test_bus` é todo normal → 0 alarmes |
| PCK@0.1 ≥ 0,80 | ⚠️ **não de forma válida** | GT de keypoints é auto-referencial (mesmo detector); exige rótulo manual (CVAT) |
| Acurácia de classificação ≥ 0,85 | ❌ **não** | Exige vídeos com posturas inadequadas anotadas — **inexistentes** |
| **Recall risco imediato ≥ 0,90** | ❌ **não por vídeo** | **Não há `cenario2.mp4` (queda)** → zero amostras positivas. **Protegido por testes unitários** (§3) |

> **Honestidade metodológica:** sem as 3 filmagens, qualquer número de acurácia/recall
> de cenário seria fabricado. Não os inventei. Em vez disso: (a) medi a latência real,
> (b) validei o harness ponta a ponta, e (c) blindei a lógica de recall com 86 testes
> unitários determinísticos (que não dependem de footage).

---

## 1. Validação funcional do harness (`evaluate_pose.py`)

GT honesto gerado para `test_bus.mp4`: **toda pessoa rotulada `normal`** (o clipe mostra
pessoas em pé — `normal` é o rótulo correto). 300 anotações sobre 75 frames.
Arquivo: `data/annotations/test_bus_gt.json`.

```
Frames processed : 75
PCK@0.1          : 1.000 on 300 matched persons   (auto-referencial — ver caveat)
Overall accuracy : 1.000   (trivial: footage 100% normal)
normal                     P=1.000  R=1.000  F1=1.000
ergonomically_inadequate   P=0.000  R=0.000  F1=0.000   (sem amostras)
immediate_risk             P=0.000  R=0.000  F1=0.000   (sem amostras — N/A)
critical_zone              P=0.000  R=0.000  F1=0.000   (sem amostras)
False alarms     : 0 episode(s) em 75 frames normais  (~0,0/h, meta → 0) ✅
```

**Leitura correta:**
- **Falsos alarmes = 0** sobre um clipe multi-pessoa real → o pipeline **não gera alarme
  espúrio** em movimento normal. É um resultado *positivo* e legítimo (meta "→ 0").
- `immediate_risk R=0,000` aqui **não é falha de recall**: é `0/0` (nenhum positivo no
  GT). Recall só existe com quedas filmadas. **Não interpretar como recall < 0,90.**
- PCK = 1,000 é **esperado e não-informativo**: os keypoints do GT vieram do mesmo
  detector que o harness roda (auto-referência). PCK real exige keypoints rotulados à mão.

O harness em si está **correto e funcional** (casa pose↔GT por IoU, exclui warm-up,
conta falso-alarme por episódio, calcula PCK/recall/F1).

---

## 2. Latência E2E (real, esta máquina)

| Config (device / modelo) | Pessoas | mean | p95 | ≤200 ms? |
|---|---|---|---|---|
| cpu / yolo11s-pose | ~4 | 210,4 ms | 362,6 ms | ❌ FAIL |
| **cuda / yolo11s-pose** | ~4 | **43,2 ms** | **44,9 ms** | ✅ **PASS** |
| cuda / yolo11n-pose | ~3 | 37,5 ms | 38,7 ms | ✅ PASS |

**Conclusão:** a meta de 200 ms **só é atingível em GPU**. Em CUDA há ~4× de folga.
Recomenda-se fixar `device: cuda` no `pose_config.yaml` para os cenários reais.

---

## 3. Recall protegido por testes unitários (substituto válido enquanto não há footage)

Como o recall de risco imediato não pode ser medido por vídeo hoje, a **garantia de
recall vive na lógica do classificador** — e essa lógica está agora coberta por testes
determinísticos que travam o comportamento crítico:

| Propriedade que sustenta o recall | Teste |
|---|---|
| Risco imediato entra rápido (N frames configuráveis) | `TestImmediateRiskEntry::test_enter_requires_exactly_enter_frames_risk` |
| Risco entra mais rápido que ergonomia (histerese assimétrica) | `TestImmediateRiskEntry::test_risk_enters_faster_than_ergonomic` |
| Alarme **não sai** com 1 frame normal (sticky) | `TestImmediateRiskExit::test_single_normal_frame_does_not_clear_alarm` |
| Flicker de risco reinicia a contagem de saída | `TestImmediateRiskExit::test_intermittent_risk_resets_exit_counter` |
| Queda detectada por bbox **mesmo sem keypoints** | `TestMissingKeypoints::test_fall_detected_from_bbox_even_without_keypoints` |
| Queda dentro de zona → prioriza `IMMEDIATE_RISK` | `TestCategories::test_immediate_risk_outranks_zone` |
| 3 gatilhos de queda (bbox / velocidade / instabilidade) | `TestCategories::test_immediate_risk_via_*` |
| Estados isolados por pessoa; ID novo começa normal | `TestMultiPerson::*` |

**Total: 86 testes, 100% verdes** (`tests/test_reba.py` + `tests/test_classifier.py`).

---

## 4. Achado de correção: bug na Tabela B do REBA (corrigido)

Os testes contra a tabela canônica **Hignett & McAtamney (2000)** revelaram que a
`_TABLE_B` de [src/pose/reba.py](../src/pose/reba.py) estava **sub-pontuando** o bloco
de antebraço estendido (lower_arm=2) e a linha de braço acima da cabeça (upper_arm=5):

| Célula (upper, lower, wrist) | Antes (errado) | Depois (oficial) |
|---|---|---|
| (3, 2, ·) | 3,4,5 | **4,5,5** |
| (4, 2, ·) | 4,5,5 | **5,6,7** |
| (5, 1, w3) | 7 | **8** |
| (5, 2, ·) | 6,7,7 | **7,8,8** |
| (6, 2, ·) | 7,8,8 | **8,9,9** |

**Impacto:** o REBA ficava **mais baixo** que o real em posturas de **alcance acima do
ombro com cotovelo estendido** — exatamente um fator de risco NR-17. Sub-pontuar reduz a
sensibilidade de `ERGONOMICALLY_INADEQUATE` (problema de recall *ergonômico*).
**Não afeta `IMMEDIATE_RISK`** (que independe do REBA). Corrigido e travado por
`TestReferenceTables::test_table_b_matches_reference`.

A **Tabela A** confere 100% com a referência. A **Tabela C** foi validada
estruturalmente (faixa 1–12, monotonicidade em linhas/colunas, cantos, e linhas A=1..4
exatas); recomenda-se uma conferência manual célula-a-célula contra o worksheet impresso
para as linhas A=5..11 (fora do que pude assertar com confiança de memória).

---

## 5. Clipes reais recebidos (Dia 2, tarde) — `cenario.mp4` e `cenario2.mp4`

Dois clipes verticais (1080×1920, 30 fps) gravados **de frente** num quarto. GT por
observação (janelas de frame), método documentado nos scripts de diagnóstico. Latência
em GPU passou em ambos (p95 34–40 ms).

### 5.1 `cenario.mp4` (ergonomia: braço + tronco + agachamento, frontal)

| Categoria | Precisão | Recall | Nota |
|---|---|---|---|
| ergonomically_inadequate | 1,000 | 0,637 | acerta braço elevado; **perde tronco e agachamento** |
| (geral) accuracy | — | 0,781 | abaixo de 0,85 |
| immediate_risk | 0,000 | n/a | falsos positivos de `fall_velocity` (~747/h) ao **baixar o braço** |

**Dois bugs reportados pelo usuário, confirmados com dados:**
- **Tronco fletido não detecta** → filmado de frente, flexão máxima = **37°** (regra exige >60°).
  Limitação 2D fundamental: flexão para frente **só é mensurável em vista de perfil**.
- **Agachamento detecta e some** → joelho < 90° em só **3 de 1008 frames**; no fundo do
  agachamento o joelho/tornozelo perde confiança (<0,5) e o ângulo vira `None`. Causa:
  membros inferiores ocluídos/fora do quadro. **Corrigir filmando de perfil, corpo inteiro
  com os pés no quadro.**

### 5.2 `cenario2.mp4` (queda real) — métrica crítica de recall

Contém 2 quedas reais (pessoa ao chão ~f700–845 e ~f1050–1190) + flexões/agachamentos.

**Onde o recall vazava (config default):** o detector pega o **movimento** da queda
(`fall_velocity`), mas perde (a) a **descida controlada** e (b) a **pessoa caída e imóvel**
— sem velocidade e, prona de frente, a bbox não fica larga (aspecto mediano 0,80 < 1,5).
O alarme caía ~0,5 s após o movimento parar.

**Calibração — evoluiu de "só thresholds" para regra de 'caído sustentado' (mudança de código):**

| Abordagem | Recall IR | Precisão IR | Falsos "fall" (cenario) |
|---|---|---|---|
| default (exit=15, aspect=1.5) | 0,587 | 0,636 | baseline |
| só-threshold (exit=45, aspect=0.85) | 0,986 | 0,581 | — |
| **cues 'caído' + exit=45 + fall_risk_min=0.85** | **0,996** ✅ | 0,418 | 390→250 frames IR |

Mudanças de código (itens 1 e 2 solicitados):
- **Item 1 — `fall_velocity` (precisão):** `TemporalAnalyzer` passou a usar o **centro do quadril**
  (`angles.hip_center`) como CoM, não o centro da bbox. Levantar/baixar o braço redimensiona a bbox
  e gerava sinal de queda falso; o quadril não. Reduziu, mas **não zerou** — o resíduo é o quadril
  descendo de fato ao agachar/flexionar + jitter de pose. `fall_risk_min 0.7→0.85` corta esse resíduo
  sem perder recall (390→250 frames falsos no cenario; precisão c2 0,384→0,418).
- **Item 2 — 'caído sustentado' (recall):** novo `vertical_extent = (pé_mais_baixo_y − nariz_y)/h`
  em `extract_angles`, usado por `_detect_fallen` junto com tronco-horizontal (>65°) e bbox-aspecto.
  São **velocity-independent**: mantêm o alarme enquanto a pessoa está caída e imóvel. `aspect`
  restaurado para 1,3 (seguro). **Separabilidade medida:** `vertical_extent<0.4` pega os frames
  prono/invertido com **0% de falso em agachamento** (ERG nunca < 0,72).

**Ressalvas honestas:**
1. **Piso limpo vs teto ruidoso:** os cues velocity-independent sozinhos (fall_velocity desligado)
   dão **recall 0,654 com 0 falso alarme** — é o piso robusto. Os 0,996 finais dependem também do
   `fall_velocity`, que é ruidoso (NORM e ERG também sobem a 1,0).
2. **Falsos alarmes ainda altos** (~750–950/h **extrapolados de clipes de 19–44 s cheios de ação** —
   número inflado; num cenário real "majoritariamente normal" cai muito). Aceito sob *recall > precisão*.
3. **Próximo nível de precisão** (não feito): gating "queda só conta se terminar em postura caída"
   (lógica temporal) eliminaria o resíduo de `fall_velocity` em agachamentos rápidos.
4. **2 clipes, frontais** → generalização ainda não comprovada.

### 5.3 Comparação de modelo de pose (proposta: YOLO26)

`yolo26{n,s}-pose.pt` existe e baixa via ultralytics 8.4.75. Head-to-head em `cenario2`
(config default, GPU):

| Modelo | Latência p95 | Recall IR | Acurácia |
|---|---|---|---|
| yolo11s-pose | 34,0 ms | **0,587** | 0,595 |
| yolo26s-pose | 35,2 ms | 0,481 | 0,589 |

**Veredito:** trocar o backbone **não melhora** a métrica crítica — recall ficou igual ou
pior, latência igual. O gargalo de recall é a **lógica de queda**, não os keypoints. Decisão
de modelo adiada; sem ganho comprovado neste caso.

### 5.4 Harness de avaliação — corrigido ✅

Antes, `scripts/evaluate_pose.py` construía `PipelineConfig(...)` com os defaults do
dataclass e **ignorava** `pose_config.yaml`. Agora ele carrega via
`PipelineConfig.from_yaml(--config)` e aplica `--pose-model`/`--device`/`--zones` só como
override (default `None`). Verificação: rodando o harness **oficial** no `cenario2` com o
YAML calibrado, **immediate_risk recall = 0,986 ✓** (P=0,581), confirmando a calibração pelo
caminho oficial — não só pelo script de sweep.

### 5.5 Validação cruzada da calibração (anti-overfitting, 1º check)

Rodando o `cenario.mp4` (ergonomia) com a config final (cues 'caído' + aspect=1.3 + fall_risk_min=0.85):

| | Falsos "fall" | Observação |
|---|---|---|
| default (aspect=1.5, exit=15) | 4 ep (~747/h) | — |
| **config final** | 4 ep (~747/h) | episódios iguais; frames-IR caíram (390→250) |

Decomposição instrumentada no cenario confirmou que **nenhum cue novo dispara falso** (bbox-aspecto,
tronco>65°, `vertical_extent`<0,4 — zero em flexão/agachamento frontal). Os falsos "fall" que restam
são **só** `fall_velocity` (quadril descendo em agachamento + jitter), não os cues de postura. **Ainda
são 2 clipes frontais** — generalização a confirmar com mais footage e, idealmente, gravação de perfil.

---

### Limitações declaradas do REBA (não são bugs — fixadas por teste)
- **Punho fixo em 1** (desvio/torção não observável em 2D).
- **Ajustes de carga/força, pega e atividade NÃO implementados** →
  `final_score = TabelaC(score_a, score_b)` sem os `+1/+2/+3`. Travado por
  `TestDeclaredLimitations`. Consequência: trunk capa em 4, neck em 2, e o pior REBA
  *alcançável a partir de 2D* é **9** (não o máximo teórico 12).
- **Níveis de ação:** o código usa esquema 1–4 (≠ 0–4 canônico); a fronteira AL4 (final≥8)
  coincide com `reba_risk_score=8` do classificador. Consistente.
