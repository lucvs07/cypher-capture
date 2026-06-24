# Relatório Consolidado — Dia 2 (23/06/2026)

**Autor:** Claude Code (Opus 4.8) · **Checkpoint de mentoria:** quinta 25/06/2026
**Foco do dia:** validação numerada das metas da Fase 2 e **proteção do recall ≥ 0,90**
por testes unitários. (LGPD/privacidade fora de escopo nesta execução, por instrução.)

---

## 1. Status do Dia 1

Detalhe completo em [auditoria-dia1-status.md](auditoria-dia1-status.md). Resumo:

| Entregável Dia 1 | Status | Nota |
|---|---|---|
| CLI `run_pipeline.py` | ✅ completo | Roda headless, `--output`, HUD de latência, sink de eventos. Falta só `--device`. |
| 3 vídeos de cenário | ❌ **ausentes** | Só `test_bus.mp4` (demo). **Bloqueador nº 1.** |
| Latência E2E medida | ⚠️ não no Dia 1 | Medida agora (Dia 2): CPU falha, **GPU passa**. |
| `zones.json` | ❌ ausente | Pipeline não quebra; criar com `annotate_zones.py`. |
| Pesos (`exp.pt`, pose) | ✅ presentes | `exp.pt` **sem classes `no_*`** → violação de EPI nunca dispara. |

> **Gap dominante:** o Dia 1 entregou *software*, não *execução*. Sem as 3 filmagens,
> as métricas de **classificação** dos cenários (acurácia, recall) não existem. Isso é o
> que a mentoria mais vai cobrar e precisa de gravação presencial.

---

## 2. Testes unitários

| Arquivo | Testes | Passam | Falham |
|---|---|---|---|
| [tests/test_reba.py](../tests/test_reba.py) | 62 | 62 | 0 |
| [tests/test_classifier.py](../tests/test_classifier.py) | 24 | 24 | 0 |
| **Total** | **86** | **86** | **0** |

Comando: `python -m pytest tests/ -q` → `86 passed in ~1,7s` (sem GPU/modelo — pura lógica).

**Cobertura:**
- **REBA:** tabelas A/B/C contra Hignett & McAtamney; sub-scores de cada articulação com
  thresholds de fronteira; combinações aleatórias (seed fixa) na faixa 1–12; casos extremos
  (neutro→1, pior alcançável→9/AL4, ângulos fora de faixa não quebram); limitações declaradas
  (punho=1; sem ajuste de carga/atividade).
- **Classificador:** histerese assimétrica (entrada rápida de risco, saída lenta/sticky,
  flicker reinicia saída, oscilação nunca confirma); as 4 categorias e a ordem de prioridade;
  fronteira REBA 7 vs 8; isolamento multi-pessoa e ID novo; keypoints ausentes; `sustained_seconds`.

**Falha encontrada e resolvida:** 2 testes falharam inicialmente isolando um **erro de
transcrição na Tabela B do REBA** (sub-pontuação de alcance acima do ombro / cotovelo
estendido). Corrigido em `src/pose/reba.py` para os valores canônicos; suíte 100% verde.
Detalhe em [metricas-dia2.md](metricas-dia2.md) §4.

---

## 3. Métricas medidas

| Métrica | Meta | Resultado Dia 2 | Status |
|---|---|---|---|
| Latência E2E (cuda / s) | ≤ 200 ms | p95 **44,9 ms** | ✅ PASS |
| Latência E2E (cpu / s) | ≤ 200 ms | p95 362,6 ms | ❌ FAIL (→ usar GPU) |
| Falsos alarmes/hora | → 0 | **0** em clipe normal multi-pessoa | ✅ |
| PCK@0.1 | ≥ 0,80 | 1,000 (auto-referencial — **inválido**) | ⚠️ N/A |
| Acurácia postural | ≥ 0,85 | — sem footage de cenário | ❌ não medível |
| **Recall risco imediato** | **≥ 0,90** | — sem footage de queda | ❌ não medível por vídeo |

**Quais atingem a meta:** latência (em GPU) e falso-alarme. **Quais ficam de fora:**
acurácia e recall — não por estarem baixos, mas porque **faltam os vídeos** que contêm os
eventos. PCK exige rótulo manual (CVAT) para ser válido. Tabela e caveats em
[metricas-dia2.md](metricas-dia2.md).

---

## 4. Calibração de recall

**Não houve iteração de calibração** porque o recall não é mensurável sem `cenario2.mp4`.
Em vez de calibrar contra um número fabricado, o recall foi **protegido na lógica** e os
"botões" de calibração ficam documentados para quando o vídeo existir:

| Sintoma futuro (com footage) | Threshold a ajustar (`pose_config.yaml`) |
|---|---|
| Queda lateral não detectada | `fallen_aspect_ratio` (1,5 → menor) |
| Queda frontal/para a câmera não detectada | `fall_risk_min` (0,7 → menor) — bbox não alarga |
| Perda de equilíbrio não detectada | `trunk_rotation_max` (0,35 → menor) |
| Risco demora a entrar | `enter_frames_risk` (2 → 1) |
| Alarme sai cedo demais | `exit_frames` (15 → maior) |

Os testes em `test_classifier.py` garantem que **mexer nesses valores produz exatamente o
efeito esperado** (ex.: `test_custom_threshold_is_honoured`, `test_enter_requires_exactly_
enter_frames_risk`), então a calibração do Dia 3+ será dirigida por dados, sem regressão silenciosa.

**Risco residual conhecido:** quedas **em direção à câmera** não alargam a bbox →
dependem do `fall_risk` temporal (heurística não validada). Filmar o Cenário 2 **de perfil**
(plano-testes-dia1.md §2a) e incluir 1 queda frontal para exercitar esse caminho.

---

## 5. Recomendações para o Dia 3

**Bloqueadores a destravar (ordem):**
1. 🔴 **Gravar os 3 cenários** (equipe presente) — protocolo pronto em
   [plano-testes-dia1.md](plano-testes-dia1.md) §2a. Cenário 2 **de perfil**, 2–3 quedas
   + 1 frontal. Sem isso não há recall/acurácia para o checkpoint.
2. 🔴 **Fixar `device: cuda`** no `pose_config.yaml` (latência só passa em GPU).
3. 🟠 Após gravar: `annotate_gt.py` → `evaluate_pose.py` por cenário; calibrar recall até ≥ 0,90.

**A documentar no README:**
- Seção Fase 2: como rodar `run_pipeline.py`, anotar zonas, rodar `pytest`, rodar `evaluate_pose.py`.
- Tabela de latência cpu vs cuda e a decisão de hardware.
- Limitações 2D declaradas do REBA (punho fixo, sem ajuste de carga/atividade, pior REBA = 9).

**Riscos residuais / ajustes pós-mentoria:**
- `exp.pt` **sem classes `no_*`** → o cruzamento EPI↔postura anunciado **não dispara**.
  Decidir: re-treinar com violações, ou inferir violação por ausência. (Não bloqueia pose/queda/zona.)
- Tabela C do REBA: conferência manual célula-a-célula das linhas A=5..11 contra o worksheet.
- `run_pipeline.py`: adicionar `--device`.
- Queda frontal (bbox não alarga): validar o `fall_risk` temporal com footage real.

**Entregue hoje:** 86 testes unitários (verdes), bug do REBA corrigido, latência E2E real
medida (GPU passa), harness de avaliação validado ponta a ponta, e 3 documentos
(auditoria, métricas, este relatório). **Pendente de footage:** acurácia e recall de cenário.
