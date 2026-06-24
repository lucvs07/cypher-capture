# Status Pré-Mentoria — Dia 3 (24/06/2026)

**Checkpoint:** Segunda Mentoria (Fernando V. Marcolina e Wendel de Almeida Passos) — quinta 25/06/2026.
**Escopo consolidado:** Sprint 1 + Sprint 2 da Fase 2.
**Verificado por:** Claude Code (Opus 4.8), execução real no ambiente (`venv` Python 3.10, CUDA disponível, `torch 2.6.0+cu124`).
**Commit base:** `fe4dc54` (branch `main`, working tree limpo).

> Este documento é a **fonte autoritativa** do estado pré-mentoria. Onde houver divergência
> com [relatorio-dia2.md](relatorio-dia2.md) §3–5 (escrito antes de as filmagens chegarem,
> quando recall/acurácia ainda eram "não mensuráveis"), **vale este documento**: os clipes
> `cenario.mp4` e `cenario2.mp4` chegaram na tarde do Dia 2, e o recall de risco imediato
> **passou a ser medido em vídeo** — ver [metricas-dia2.md](metricas-dia2.md) §5.

---

## Seção 1 — Resumo executivo

A Fase 2 (estimação de pose e análise comportamental) foi implementada, testada e validada com
footage real. O pipeline unificado EPI+pose roda em tempo real com latência E2E de **p95 44,9 ms
em GPU** (CPU estoura o orçamento — 362 ms). O **recall de detecção de risco imediato atinge 0,986**
pelo harness oficial em `cenario2.mp4` (0,996 no sweep de calibração), **excedendo a meta de 0,90**.
A lógica crítica (motor REBA e histerese do classificador) está coberta por **89 testes unitários
determinísticos, todos verdes**, e um bug de transcrição da Tabela B do REBA foi encontrado por
esses testes e corrigido. A documentação técnica (README em português com seção Fase 2, taxonomia,
três relatórios de auditoria) está completa. **Pendências honestas:** a acurácia postural ficou em
0,78 (abaixo de 0,85) por limitação fundamental de visão 2D frontal (flexão de tronco e agachamento
não são mensuráveis de frente), o terceiro cenário (zona crítica) ainda não foi filmado, e a
precisão de risco imediato é baixa (0,42–0,58) por opção deliberada de priorizar recall sobre
precisão em contexto de segurança.

---

## Seção 2 — O que foi entregue (Dias 1–3)

### Dia 1 — software e scaffolding
- ✅ **CLI do pipeline unificado** (`src/run_pipeline.py`): roda headless, `--output`, HUD de
  latência por etapa, sink de eventos por severidade, e flag `--device` (adicionada no Dia 2).
- ❌→⚠️ **3 vídeos de validação:** o Dia 1 entregou *software*, não *execução presencial*. As
  filmagens chegaram **no Dia 2**: `cenario.mp4` (ergonomia, 33,6 s) e `cenario2.mp4`
  (queda + ergonomia, 44,4 s). **O terceiro (zona crítica) não foi filmado.**
- ⚠️→✅ **Latência medida:** não no Dia 1; medida no Dia 2 — GPU p95 **44,9 ms** PASSA, CPU
  362 ms FALHA.

### Dia 2 — testes, métricas e calibração
- ✅ **Testes unitários de REBA** (`tests/test_reba.py`): tabelas A/B/C vs Hignett & McAtamney.
- ✅ **Testes unitários de histerese** (`tests/test_classifier.py`): entrada rápida de risco,
  saída sticky, flicker, isolamento multi-pessoa, keypoints ausentes.
- ✅ **Métricas de Fase 2 medidas:** latência (real), PCK (auto-referencial — caveat), recall e
  acurácia (sobre os 2 clipes reais), falso-alarme.
- ✅ **Recall de risco imediato ≥ 0,90:** **0,986** (harness oficial) / **0,996** (sweep) em
  `cenario2.mp4`, após calibração (regra de "caído sustentado" + `exit_frames` 15→45 +
  `fall_risk_min` 0,7→0,85).
- ✅ **Bug corrigido:** Tabela B do REBA sub-pontuava alcance acima do ombro/cotovelo estendido.

### Dia 3 — consolidação e apresentação
- ✅ **README atualizado** com seção Fase 2 completa (como rodar CLI, anotar zonas, rodar testes,
  rodar `evaluate_pose.py`, tabela de latência CPU vs GPU, limitações 2D do REBA).
- ✅ **Documentação consolidada:** este documento + esqueleto de apresentação + checklist de
  prontidão.
- ✅ **Esqueleto de apresentação para Claude Design:** [apresentacao-esqueleto.md](apresentacao-esqueleto.md).

---

## Seção 3 — Métricas finais

| Métrica | Meta | Obtido | Status |
|---|---|---|---|
| PCK@0.1 | ≥ 0,80 | 1,000 — **auto-referencial** (GT veio do mesmo detector) | ⚠️ N/A (exige rótulo manual CVAT) |
| Acurácia postural | ≥ 0,85 | **0,78** (`cenario.mp4`, vista frontal) | ❌ (limitação 2D — ver Seção 4) |
| Recall risco imediato | ≥ 0,90 | **0,986** (harness oficial) / 0,996 (sweep) em `cenario2.mp4` | ✅ |
| Latência E2E | ≤ 200 ms | **p95 44,9 ms** (GPU) · 362 ms (CPU) | ✅ em GPU / ❌ em CPU |
| Falsos alarmes/hora | → 0 | **0** em clipe normal multi-pessoa | ✅ (clipe normal) |
| Precisão risco imediato | (sem meta) | 0,42–0,58 | ⚠️ baixa — *recall > precisão* por opção |

**Notas de honestidade metodológica:**
- **PCK = 1,000 não é informativo:** os keypoints de ground-truth vieram do próprio detector
  (auto-referência). Um PCK independente exige keypoints rotulados manualmente (CVAT).
- **Falso-alarme:** 0 em clipe normal. Em clipes de ação curtos (19–44 s, cheios de
  flexões/agachamentos) a extrapolação por hora chega a ~750–950/h — **número inflado** pela
  natureza do clipe; num cenário "majoritariamente normal" cai muito. Aceito sob a política de
  priorizar recall.

---

## Seção 4 — Riscos residuais para a mentoria

1. **Acurácia postural 0,78 < 0,85 — limitação fundamental de 2D, não bug.**
   Filmagem frontal: a flexão de tronco para a frente mede no máximo 37° (regra exige > 60°) e o
   agachamento profundo oclui joelho/tornozelo (confiança < 0,5 → ângulo `None`).
   **Plano de ação:** refilmar o cenário de ergonomia **de perfil**, corpo inteiro com os pés no
   quadro. A regra e os thresholds já estão prontos; é uma questão de ângulo de câmera.

2. **Zona crítica fora do escopo desta demo (decisão de equipe).** A funcionalidade está
   implementada (`src/zones/`, `annotate_zones.py`), mas **não será demonstrada nesta mentoria**;
   o cenário 3 não foi filmado e `config/zones/zones.json` não existe (só `zones.example.json`).
   O pipeline não quebra sem zonas (`0 zone(s) loaded`). **Pós-mentoria:** filmar o cenário,
   depois `annotate_zones.py --source ... --output config/zones/zones.json`.

3. **Precisão de risco imediato baixa (0,42–0,58).** Decisão deliberada: em segurança, um falso
   negativo (não detectar queda) é inaceitável; um falso alarme é tolerável. O próximo nível de
   precisão (gating temporal — "queda só conta se terminar em postura caída") está identificado mas
   não implementado (congelado até pós-mentoria).

4. **Decisão técnica pendente — CPU vs GPU.** A latência só passa em GPU. É preciso confirmar o
   alvo de hardware da Metaindústria com a SPI. **Recomendação:** fixar `device: cuda` no
   `pose_config.yaml` (hoje o default é `cpu`).

5. **`exp.pt` não tem classes `no_*`.** O detector de EPI só reconhece *presença* de equipamento,
   não *violação*. O cruzamento EPI↔postura anunciado **não dispara** com este peso. Não bloqueia
   pose/queda/zona. **Plano:** re-treinar com violações ou inferir violação por ausência.

6. **Generalização não comprovada:** apenas 2 clipes, ambos frontais, num quarto. A calibração de
   recall ainda não foi validada em footage variado nem em gravação de perfil.

7. **Tabela C do REBA:** validada estruturalmente; recomenda-se conferência manual célula-a-célula
   das linhas A=5..11 contra o worksheet impresso.

---

## Seção 5 — Estrutura de apresentação (Figma / Claude Design)

A apresentação tem 5 seções. O conteúdo de cada uma vem das auditorias abaixo:

1. **Apresentação do grupo** (nomes, RMs, papéis, fotos) — preencher papéis da equipe
   (placeholders em [apresentacao-esqueleto.md](apresentacao-esqueleto.md) §1).
2. **Contextualização** (dor, solução, entrega de setembro) — de [relatorio-dia2.md](relatorio-dia2.md)
   e da visão do produto (Cypher — Industrial Vision AI).
3. **Demonstração prática** (2 cenários Fase 1+2: ergonomia e queda) — de
   [metricas-dia2.md](metricas-dia2.md) §5; vídeos anotados em `outputs/captures/`
   (`cenario_annot.mp4`, `cenario2_annot.mp4`). **Zona crítica fora do escopo desta demo.**
4. **Materiais e métodos** (stack, ambientes, normas, hardware) — README + este documento.
5. **Cronograma** (onde estão, o que fizeram, o que falta) — **desta** `status-pre-mentoria.md`
   (Seção 2) e do esqueleto §5.

O esqueleto completo, pronto para alimentar o Claude Design, está em
[apresentacao-esqueleto.md](apresentacao-esqueleto.md).
