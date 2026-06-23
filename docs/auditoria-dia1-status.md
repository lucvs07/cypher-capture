# Auditoria de Status — Dia 1 (verificação no início do Dia 2)

**Data:** 23/06/2026
**Verificado por:** Claude Code (Opus 4.8), execução real no ambiente (`venv` Python 3.10, CUDA disponível)
**Commit base:** `08f1b63` (branch `main`, working tree limpo no início)

## 0. Achado central (ler primeiro)

> **O Dia 1 entregou o *código/scaffolding*, mas NÃO a *execução*.** Tudo o que é
> artefato de software — CLI do pipeline, anotador de ground-truth, instrumentação
> de latência, zona de exemplo — existe e funciona. O que **não** existe é o
> produto da execução presencial do Dia 1: **os 3 vídeos de cenário, a tabela de
> latência preenchida e a primeira medição de recall**. Não há `cenario1/2/3.mp4`,
> não há anotações, não há `zones.json`. O único vídeo no repo é `test_bus.mp4`
> (clipe-demo genérico da Ultralytics, 75 frames).

Consequência direta para o Dia 2: **as métricas de classificação dos cenários da
Fase 2 (acurácia postural, recall de risco imediato, PCK real) não são mensuráveis
hoje** — faltam as filmagens. O que **é** mensurável e foi feito: latência E2E real
(nesta máquina), validação funcional do harness de avaliação, e **proteção do recall
por testes unitários** (ver [metricas-dia2.md](metricas-dia2.md) e
[relatorio-dia2.md](relatorio-dia2.md)).

---

## 1. CLI do UnifiedPipeline — ✅ completo (com 1 ressalva)

| Critério | Status | Evidência |
|---|---|---|
| `src/run_pipeline.py` existe | ✅ | [src/run_pipeline.py](../src/run_pipeline.py) (169 linhas) |
| Roda com `--source <vídeo/0>` | ✅ | Executado headless em `test_bus.mp4`: 75 frames processados sem erro |
| Imprime FPS e latência em tempo real | ✅ | HUD por frame em `\r` (pose/epi/analysis/overlay/stride) + resumo final mean/p50/p95/max |
| Aceita `--output` (salva vídeo anotado) | ✅ | Gerou `outputs/captures/test_bus_annot.mp4` (1,8 MB) |
| Aceita `--no-display` (headless) | ✅ | Rodou sem GUI; flags extras: `--min-severity`, `--max-frames`, `--show-epi` |
| Surfaceia eventos no terminal | ✅ | Sink de eventos por `severity ≥ --min-severity` |

**Ressalva (melhoria, não bug):** `run_pipeline.py` **não tem flag `--device`** — o
device vem só do `pose_config.yaml` (hoje `cpu`). Para medir em GPU é preciso editar o
YAML ou usar `evaluate_pose.py --device cuda` (que tem a flag). Recomendação: adicionar
`--device` ao `run_pipeline.py` para alternar cpu/cuda sem editar config.

---

## 2. Vídeos de validação — ❌ ausentes

| Esperado | Presente? |
|---|---|
| `data/validation/cenario1.mp4` (ergonomia) | ❌ não existe |
| `data/validation/cenario2.mp4` (risco imediato) | ❌ não existe |
| `data/validation/cenario3.mp4` (zona crítica) | ❌ não existe |
| `data/validation/test_bus.mp4` | ✅ 75 frames, 15 fps, 810×1080, ~4 pessoas/frame (demo Ultralytics, **não é cenário Fase 2**) |

**Causa:** a gravação presencial do Dia 1 não foi realizada (ou os arquivos não foram
commitados). Não há erro de software impedindo — o protocolo de gravação está pronto em
[plano-testes-dia1.md](plano-testes-dia1.md) §2a. **Este é o bloqueador nº 1 do checkpoint
de quinta** e precisa de equipe presente para filmar.

---

## 3. Latência E2E — ⚠️ não medida no Dia 1; **medida no Dia 2 (real)**

Não há artefato de latência do Dia 1 no repo. Medi agora, de verdade, sobre
`test_bus.mp4` (~4 pessoas/frame — carga representativa), via `evaluate_pose.py`:

| Config (device / modelo pose) | Pessoas | mean E2E | p95 E2E | ≤200 ms? |
|---|---|---|---|---|
| **cpu** / yolo11s-pose | ~4 | 210,4 ms | 362,6 ms | ❌ FAIL |
| **cuda** / yolo11s-pose | ~4 | 43,2 ms | 44,9 ms | ✅ PASS |
| **cuda** / yolo11n-pose | ~3 | 37,5 ms | 38,7 ms | ✅ PASS |

- **Device:** CUDA disponível (`torch 2.6.0+cu124`, `cuda=True`). O default do YAML é `cpu`.
- **Conclusão de hardware:** a meta de **≤200 ms é inatingível em CPU** com ~4 pessoas, e
  **folgada em GPU** (p95 ~45 ms, 4× abaixo do orçamento). **Decisão recomendada:
  rodar em `device: cuda`.** `yolo11n-pose` dá ~15% a mais de margem, mas detectou
  menos pessoas (3 vs 4/frame) — trade-off de recall de detecção a validar nos cenários reais.
- Warm-up (1º frame, carga de modelo) corretamente excluído das estatísticas pelo harness.

> Comando exato usado:
> `python scripts/evaluate_pose.py --annotations data/annotations/test_bus_gt.json --video data/validation/test_bus.mp4 --device {cpu|cuda} [--pose-model yolo11n-pose.pt]`

---

## 4. Zonas críticas — ❌ `zones.json` ausente

| Item | Status |
|---|---|
| `config/zones/zones.json` | ❌ ausente (só `zones.example.json`) |
| Zona desenhada p/ Cenário 3 | ❌ inexistente |
| Pipeline quebra sem zonas? | ✅ não — `0 zone(s) loaded`, segue normal |
| Ferramenta de criação | ✅ `scripts/annotate_zones.py` pronta |

**Como criar** (após ter o `cenario3.mp4`):
```bash
python scripts/annotate_zones.py --source data/validation/cenario3.mp4 --output config/zones/zones.json
# OU, sem footage ainda, partir do exemplo e ajustar vértices à resolução:
cp config/zones/zones.example.json config/zones/zones.json
```

---

## 5. Pesos de modelos — ✅ presentes

| Peso | Status |
|---|---|
| `models/exp.pt` (EPI) | ✅ 52 MB, commitado |
| `yolo11s-pose.pt` (pose) | ✅ 20 MB, na raiz do projeto |
| `yolo11n-pose.pt` (pose, baixado no Dia 2) | ✅ 6 MB |

**Achado crítico confirmado (auditoria #5):** `exp.pt` tem **7 classes** —
`{GLOVE, Goggles, HELMET, MASK, Person, VEST, boots}` — **sem nenhuma classe `no_*`**.
Como `DecisionEngine.EPI_VIOLATION_LABELS = {no_gloves, no_goggle, no_helmet, no_boots, none}`,
**`event.epi_violations` fica sempre vazio**: a severidade por falta de EPI (sev 2) e o
bônus §4.2 de zona+EPI **nunca disparam**. O modelo só detecta *presença* de EPI, não
*violação*. Isto **não bloqueia** pose/queda/zona, mas o cruzamento EPI↔postura anunciado
não funciona com este peso. **Correção:** treinar/obter um modelo com classes `no_*`, ou
inferir violação por ausência (ex.: "Person sem HELMET associado" → `no_helmet`).

---

## Resumo de gaps e correções rápidas

| # | Gap | Severidade | Correção |
|---|---|---|---|
| 1 | 3 vídeos de cenário ausentes | 🔴 bloqueador | Gravar (equipe presente) — protocolo pronto |
| 2 | Latência CPU > 200 ms | 🔴 alta | Usar `device: cuda` (já disponível) — resolve |
| 3 | `zones.json` ausente | 🟠 média | `annotate_zones.py` após ter `cenario3.mp4` |
| 4 | `exp.pt` sem classes `no_*` | 🟠 média | Modelo com violações, ou violação-por-ausência |
| 5 | `run_pipeline.py` sem `--device` | 🟡 baixa | Adicionar flag (1 linha argparse) |
| 6 | **Bug Tabela B do REBA** (achado no Dia 2) | 🟠 média | ✅ **corrigido** — ver [metricas-dia2.md](metricas-dia2.md) |
