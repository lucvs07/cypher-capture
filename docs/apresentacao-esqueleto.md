# Apresentação Sprint 1 + Sprint 2 — Segunda Mentoria Challenge 2026

> **Entrada para o Claude Design.** Dados reais do repositório (commit `fe4dc54`, 24/06/2026).
> Onde houver `[PREENCHER]`, a equipe completa antes de gerar o Figma.

## Seção 1 — Apresentação do Grupo

### Nomes e papéis
- **Lucas Rodrigues Grecco** (RM 558261) — Arquitetura e integração
- **Felipe Wapf Fettback** (RM 557217) — `[PREENCHER papel]`
- **Monique Ferreira Dos Anjos** (RM 558262) — `[PREENCHER papel]`
- **Rafael Augusto Oliveira Silva** (RM 555154) — `[PREENCHER papel]`
- **Ronaldo Veloso Filho** (RM 556445) — `[PREENCHER papel]`
- **Tiago Brito Nário** (RM 558248) — `[PREENCHER papel]`

### Fotos
`[Placeholder: foto de grupo ou individuais]`

---

## Seção 2 — Contextualização e Atualização do Projeto

**Título:** Cypher — Industrial Vision AI
**Visão:** Predição **proativa** de riscos em ambientes industriais, não detecção reativa de infrações.

Sistema de visão computacional que detecta em tempo real:
- **(Fase 1) EPIs** — capacete, óculos, máscara, colete, luvas, botas — **COMPLETO**
- **(Fase 2) Posturas de risco ergonômico e quedas iminentes** — **EM VALIDAÇÃO**
- **(Fase 3) Notificação inteligente** — não iniciada
- **(Fase 4) Validação industrial na Metaindústria** — planejada

### Dor
Acidentes laborais por falta de EPI ou postura ergonomicamente inadequada são evitáveis com
monitoramento contínuo. Hoje a detecção é **reativa** (após o acidente) ou **manual** (inspetor
visual).

### Solução
- Câmera + **YOLOv8** (EPI) + **YOLO11-pose + ByteTrack** (ergonomia/queda)
- Motor **REBA** para scoring postural contra **NR-17** e **NR-35**
- Classificador com histerese temporal (4 categorias) e zonas críticas demarcadas
- Notificação em tempo real (Fase 3, planejada)

### Produto em setembro
- Sistema rodando na Metaindústria com ≥ 3 câmeras
- Recall ≥ 0,90 em detecção de risco imediato — **já atingido em validação (0,986)**
- Dashboard web com eventos e histórico
- Documentação técnica e arquitetura de deploy

---

## Seção 3 — Demonstração Prática

**Formato:** vídeos anotados (`outputs/captures/`), ~30–45 s cada.

### Cenário 1 — Postura ergonomicamente inadequada (`cenario.mp4`, 33,6 s) ✅
- Operador com braço elevado acima do ombro detectado (ERG recall = 0,64)
- Score REBA + categoria `ERGONOMICALLY_INADEQUATE` sobrepostos
- Output: `outputs/captures/cenario_annot.mp4`
- ⚠️ **Caveat honesto (mostrar na mentoria como aprendizado):** flexão de tronco e agachamento
  **não** são detectados nesta filmagem **frontal** (limitação 2D — só mensuráveis de perfil).
  Acurácia geral 0,78. Plano: refilmar de perfil.

### Cenário 2 — Risco imediato / queda (`cenario2.mp4`, 44,4 s) ✅ **métrica crítica**
- 2 quedas reais (pessoa ao chão) + flexões/agachamentos
- Transição para `IMMEDIATE_RISK`; alarme mantido durante "caído e imóvel" (regra
  velocity-independent)
- **Recall = 0,986** (harness oficial) — **excede a meta de 0,90**
- Output: `outputs/captures/cenario2_annot.mp4`

### Cenário 3 — Aproximação a zona crítica ⚠️ **PENDENTE DE FILMAGEM**
- Zona desenhada com `annotate_zones.py` (amarelo = APPROACHING, vermelho = INSIDE)
- Categoria muda dinamicamente com a posição
- **Status:** vídeo não gravado e `config/zones/zones.json` ausente. Opções para a demo:
  (a) gravar antes de quinta; (b) demonstrar ao vivo com zona de exemplo ajustada à resolução.

---

## Seção 4 — Materiais e Métodos

### Stack de ferramentas
- **CV/ML:** Ultralytics YOLO (v8 para EPI, v11-pose + ByteTrack para ergonomia/queda)
- **Numérico:** NumPy, SciPy, OpenCV
- **Config:** YAML (fonte única de thresholds em `pose_config.yaml`)
- **Testes:** pytest (89 testes determinísticos)
- **Upload de capturas:** Cloudinary (Fase 1)
- **Backend / persistência / fila (Fase 3, planejado):** FastAPI, PostgreSQL, Redis, RabbitMQ
- **Deploy (Fase 4, planejado):** k3s, ONNX

### Ambientes
- **Desenvolvimento:** laptop/desktop, Python 3.10, GPU CUDA (`torch 2.6.0+cu124`)
- **Teste:** Kaggle Notebooks (GPU grátis), cloud (Railway/RunPod)
- **Produção (Fase 4):** Metaindústria (hardware a definir com a SPI)

### Linguagens
- Python (core de IA/CV) · JavaScript/React (dashboard, planejado) · SQL · YAML

### Hardware
- **Mínimo:** 1 câmera USB ou IP (RTSP)
- **Alvo:** 3–5 câmeras em paralelo, latência E2E ≤ 200 ms
- **GPU (medido nesta máquina):** CUDA, `yolo11s-pose` → **p95 44,9 ms** (4× de folga)
- **CPU:** estoura o orçamento (p95 362 ms) → **GPU é requisito** para tempo real

### Normas e padrões
- **NR-06** (EPI) · **NR-17** (ergonomia) · **NR-35** (trabalho em altura)
- **REBA** (Hignett & McAtamney, 2000) — método de scoring postural
- **LGPD** — fora do escopo desta auditoria (será tratada na arquitetura de deploy)

---

## Seção 5 — Cronograma Detalhado / Plano de Trabalho

### Onde estão agora? (Status Dia 3)

**Fase 1 (Detecção de EPIs) — COMPLETA**
- ✅ Modelo YOLO treinado (mAP@0.5 = 0,809)
- ✅ 7 classes: `GLOVE, Goggles, HELMET, MASK, Person, VEST, boots`
- ✅ Pipeline funcional (`src/detect.py`) + upload Cloudinary

**Fase 2 (Pose e Ergonomia) — EM VALIDAÇÃO**
- ✅ YOLO11-pose integrado · 17 keypoints COCO
- ✅ Motor REBA (tabelas A/B/C; bug da Tabela B encontrado por teste e corrigido)
- ✅ Classificador com 4 categorias + histerese assimétrica
- ✅ Detecção de zonas críticas (código pronto; falta footage + `zones.json`)
- ✅ Análise temporal (queda por velocidade + postura "caída" sustentada)
- ✅ CLI unificado (`src/run_pipeline.py`, com `--device`)
- ✅ **89 testes unitários** (REBA + histerese), 100% verdes
- ⚠️ Métricas validadas: **Recall IR 0,986 ✅** · Acurácia 0,78 ❌ (2D frontal) · Latência GPU
  p95 44,9 ms ✅ · PCK 1,000 (auto-referencial, inválido)

**Fase 3 (Notificação Inteligente) — NÃO INICIADA**
- [ ] Consumer RabbitMQ · [ ] Dashboard web · [ ] 2+ canais de notificação

**Fase 4 (Validação Industrial) — NÃO INICIADA**
- [ ] Deploy na Metaindústria · [ ] Testes com múltiplas câmeras

### O que já fizeram (Sprint 1 + 2)
- **Sprint 1:** implementação completa da Fase 2 (código), CLI, instrumentação de latência, footage
  de validação (2 dos 3 cenários).
- **Sprint 2:** 89 testes unitários, ground-truth anotado dos clipes, `evaluate_pose.py` executado
  com métricas reais, calibração de recall ≥ 0,90, documentação consolidada.

### O que ainda falta
**Antes da Banca Preliminar (Fase 4):**
- [ ] Refilmar ergonomia **de perfil** (subir acurácia acima de 0,85)
- [ ] Filmar cenário 3 (zona) + gerar `config/zones/zones.json`
- [ ] Documentar arquitetura de deploy · [ ] Preparar pitch (todos falam)

**Antes do NEXT (Fase 4):**
- [ ] Integração Fase 3 (notificação) · [ ] Deploy em ambiente real · [ ] Métricas em produção

**Pós-NEXT:**
- [ ] Iterações por feedback da banca · [ ] Re-treinar `exp.pt` com classes `no_*` (violação de EPI)
- [ ] Gating temporal de queda (subir precisão de IR) · [ ] Otimização ONNX

---

## Notas de Design para Claude Design

### Paleta de cores
- **Primário:** `#FF6A1E` (laranja Cypher)
- **Secundário:** `#ED145B` (rosa FIAP)
- **Neutro:** `#0a0a0f` (fundo escuro), `#e8e6e0` (texto claro)

### Tipografia
- **Headlines:** Courier New (monospace), bold, 28–32px
- **Body:** Calibri, regular, 13–14px
- **Dados/métricas:** Courier New, 11–12px

### Assets esperados
- Logo Cypher · Logo FIAP · Logo SPI Integração
- Prints de telas: vídeos anotados (`outputs/captures/*_annot.mp4`), HUD de latência
- Gráfico de série temporal do score REBA / categoria ao longo de `cenario2.mp4`

### Estrutura de slides esperada
- Slide 1: Capa (grupo + fotos)
- Slides 2–3: Contextualização (dor → solução → entrega)
- Slides 4–6: Demos (cenário 1 ergonomia, cenário 2 queda, cenário 3 zona/pendente)
- Slides 7–8: Stack técnica + normas
- Slides 9–10: Cronograma e status (Fases 1–4)
- Slide 11: Próximos passos e call-to-action

### Mensagens-chave (destacar visualmente)
- **Recall de risco imediato 0,986** — meta de 0,90 superada
- **Latência p95 45 ms em GPU** — 4× abaixo do orçamento de 200 ms
- **89 testes unitários verdes** — lógica crítica blindada; bug do REBA pego por teste
- **Honestidade metodológica:** nada de número fabricado; limitações 2D declaradas
