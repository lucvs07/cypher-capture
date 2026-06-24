# Cypher Capture — Visão Computacional Industrial

Monitoramento de segurança em tempo real para canteiros de obras, construído sobre [Ultralytics YOLO](https://docs.ultralytics.com/).

O projeto tem duas fases que compartilham a mesma base de código:

- **Fase 1 — Detecção de EPIs:** identifica equipamentos de proteção individual (presentes ou ausentes) a partir de webcam, vídeo ou imagens, e opcionalmente envia os frames de violação para o Cloudinary.
- **Fase 2 — Pose e análise comportamental:** um pipeline unificado que roda estimação de pose (YOLO11-pose + ByteTrack) junto com a detecção de EPIs no mesmo laço, calcula ângulos articulares e um escore ergonômico **REBA** completo, classifica a postura em quatro categorias com histerese temporal e sinaliza a entrada/aproximação em zonas críticas demarcadas.

Referências normativas: **NR-17** (ergonomia), **NR-35** (trabalho em altura) e **REBA** (Hignett & McAtamney, 2000). A taxonomia completa de posturas está em [docs/taxonomy.md](docs/taxonomy.md).

---

## Estrutura do projeto

```
cypher-capture/
├── config/
│   ├── construction-ppe.yaml     # dataset da Fase 1 (caminhos + nomes das classes)
│   ├── pose_config.yaml          # Fase 2: fonte única de thresholds e parâmetros
│   └── zones/                    # zones.json (gerado pelo anotador de zonas)
├── data/
│   ├── annotations/              # ground-truth JSON para avaliação
│   └── validation/               # vídeos de validação locais (fora do versionamento)
├── docs/
│   ├── taxonomy.md               # taxonomia de posturas, REBA, máquina de estados
│   ├── auditoria-dia1-status.md  # status do Dia 1
│   ├── metricas-dia2.md          # métricas medidas no Dia 2
│   └── relatorio-dia2.md         # relatório consolidado do Dia 2
├── models/
│   └── exp.pt                    # detector de EPI treinado (7 classes)
├── scripts/
│   ├── download_dataset.py       # baixa o dataset Construction-PPE
│   ├── annotate_zones.py         # desenha polígonos de zona crítica
│   ├── annotate_gt.py            # rotula a postura de referência por frame
│   ├── benchmark_pose.py         # benchmark de latência do modelo de pose
│   └── evaluate_pose.py          # métricas: PCK / acurácia / recall / latência
├── tests/
│   ├── test_reba.py              # testes do motor REBA (tabelas A/B/C)
│   └── test_classifier.py        # testes da máquina de estados e histerese
└── src/
    ├── detect.py                 # entry-point da Fase 1 (apenas EPI)
    ├── run_pipeline.py           # entry-point da Fase 2 (EPI + pose unificados)
    ├── train.py                  # treino do detector de EPI
    ├── pose/                     # estimador, ângulos, REBA, classificador, temporal
    ├── zones/                    # gerenciador e detector de zonas
    ├── fusion/                   # motor de decisão + pipeline unificado
    └── utils/                    # vídeo, visualização e upload
```

---

## Instalação

### 1. Clonar e criar o ambiente virtual

```bash
git clone git@github.com:lucvs07/cypher-capture.git
cd cypher-capture
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
```

### 2. Instalar o PyTorch

A meta de latência (≤ 200 ms ponta a ponta) só é realista em **GPU**. Escolha uma opção:

```bash
# GPU (NVIDIA, CUDA 12.x) — recomendado; necessário para cumprir o orçamento de latência
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Somente CPU (funciona, mas o pipeline unificado estoura a latência)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Confirme que a GPU está visível:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

### 3. Instalar as demais dependências

```bash
pip install -r requirements.txt
```

Os pesos de pose (`yolo11s-pose.pt`) são baixados automaticamente na primeira execução. Para pré-baixar:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11s-pose.pt')"
```

### 4. (Apenas Fase 1) Configurar o Cloudinary

Necessário só se você quiser que a Fase 1 envie os frames de violação. Copie `.env.example` para `.env` e preencha as credenciais. Caso contrário, use `--no-upload`.

> **Rode sempre os comandos a partir da raiz do projeto** — os caminhos de modelos e configs são relativos a ela.

---

## Modelos

| Modelo | Arquivo | Tarefa | Observações |
|---|---|---|---|
| Detector de EPI | `models/exp.pt` | detect | 7 classes: `GLOVE, Goggles, HELMET, MASK, Person, VEST, boots` |
| Estimador de pose | `yolo11s-pose.pt` | pose | baixado automaticamente; 17 keypoints COCO |

Para um modelo de pose mais rápido ou mais preciso, troque em `config/pose_config.yaml → pipeline.pose_model_path`: `yolo11n-pose.pt` (mais rápido), `yolo11s-pose.pt` (padrão) ou `yolo11m-pose.pt` (mais preciso).

> O modelo de EPI emite apenas classes de **presença** (não há classes `no_*` de violação). Por isso o pipeline unificado reporta os EPIs detectados, mas **não** levanta eventos de violação de EPI. Para inspecionar as classes de quaisquer pesos:
> `python -c "from ultralytics import YOLO; print(YOLO('models/exp.pt').names)"`

---

## Como rodar — Fase 2 (EPI + pose unificados)

Este é o entry-point principal. Ele lê todos os parâmetros de `config/pose_config.yaml`.

```bash
# Webcam, com janela ao vivo (use cuda para cumprir a latência)
python src/run_pipeline.py --source 0 --device cuda

# Arquivo de vídeo, salvando o vídeo anotado
python src/run_pipeline.py --source data/validation/clip.mp4 --device cuda \
  --output outputs/captures/clip_anotado.mp4

# Headless (sem janela — servidores / medição)
python src/run_pipeline.py --source data/validation/clip.mp4 --device cuda --no-display

# Com zonas críticas
python src/run_pipeline.py --source 0 --device cuda --zones config/zones/zones.json
```

| Flag | Padrão | Descrição |
|---|---|---|
| `--source` | — | índice da webcam, caminho do vídeo ou URL RTSP (obrigatório) |
| `--config` | `config/pose_config.yaml` | arquivo de configuração do pipeline |
| `--device` | do config | sobrescreve o device: `cpu` \| `cuda` \| `mps` |
| `--zones` | do config | sobrescreve o JSON de zonas críticas |
| `--output` | — | salva o vídeo anotado neste caminho |
| `--no-display` | desligado | roda sem janela (GUI) |
| `--min-severity` | `2` | imprime eventos com severidade ≥ este valor (0–4) |
| `--max-frames` | `0` | para após N frames (0 = até o fim da fonte / `q`) |
| `--show-epi` | desligado | desenha as caixas cruas do detector de EPI |

Cada frame imprime um HUD de latência por etapa (`pose | epi | analysis | overlay`), os eventos acima do limiar de severidade e, ao final, um resumo de latência com o veredito (passou/falhou) contra o orçamento de 200 ms. Pressione **`q`** para sair de uma execução com janela.

### Latência de referência

Medida sobre um clipe com ~4 pessoas, `yolo11s-pose` + `exp.pt`:

| Device | p95 E2E | Veredito (≤ 200 ms) |
|---|---|---|
| CPU | ~362 ms | ❌ FALHA |
| GPU (CUDA) | ~45 ms | ✅ PASSA |

A conclusão prática é direta: **rode em GPU** (`--device cuda` ou `pipeline.device: cuda` no YAML).

---

## Categorias de postura e detecção de queda

O classificador atribui a cada pessoa uma das quatro categorias, em ordem de prioridade:

| Categoria | Quando dispara |
|---|---|
| `immediate_risk` | queda detectada (movimento, postura horizontal/invertida ou perda de equilíbrio) |
| `critical_zone` | pessoa dentro de um polígono de zona demarcada |
| `ergonomically_inadequate` | postura de risco sustentada (REBA alto, agachamento, tronco fletido, braço elevado) |
| `normal` | nenhuma condição acima |

A máquina de estados usa **histerese assimétrica**: entrar num alarme é rápido (poucos frames), sair é lento (`exit_frames`). Isso prioriza o **recall** — em segurança, um falso negativo (não detectar uma queda) é inaceitável, enquanto um falso alarme é tolerável.

A detecção de queda combina vários sinais (operador lógico OU, priorizando recall):
- **velocidade de queda** a partir do centro do quadril (estável a movimentos de braço);
- **postura "caída" sustentada**, independente de velocidade — tronco horizontal, ou eixo cabeça→pés colapsado (`vertical_extent`) — que mantém o alarme ativo mesmo com a pessoa caída e imóvel.

Todos os limiares são configuráveis em `config/pose_config.yaml`, sem alterar o código.

---

## Como rodar — Fase 1 (apenas detecção de EPI)

```bash
# Webcam (note o prefixo "usb" aqui)
python src/detect.py --model models/exp.pt --source usb0 --no-upload

# Vídeo / imagem única / diretório de imagens
python src/detect.py --model models/exp.pt --source clip.mp4 --no-upload
python src/detect.py --model models/exp.pt --source imagem.jpg --no-upload
python src/detect.py --model models/exp.pt --source imagens/ --no-upload
```

| Flag | Padrão | Descrição |
|---|---|---|
| `--model` | — | caminho dos pesos YOLO (obrigatório) |
| `--source` | — | `usb0`, vídeo, imagem ou diretório (obrigatório) |
| `--thresh` | `0.5` | confiança mínima para exibir / sinalizar |
| `--cooldown` | `5.0` | segundos entre uploads em streams ao vivo |
| `--no-upload` | desligado | desativa o upload para o Cloudinary |

O `detect.py` sempre abre uma janela (não tem modo headless).

---

## Zonas críticas

Desenhe os polígonos de zona sobre um frame da **mesma câmera/resolução** em que você vai rodar:

```bash
python scripts/annotate_zones.py --source data/validation/clip.mp4 --output config/zones/zones.json
# clique com o botão esquerdo nos vértices → ENTER → informe nome/tipo/severidade → q para sair
```

Há um exemplo inicial em `config/zones/zones.example.json` (copie para `config/zones/zones.json` e ajuste os vértices à sua resolução).

---

## Avaliação e benchmark

**Latência do modelo de pose isolado:**

```bash
python scripts/benchmark_pose.py --source data/validation/clip.mp4 --frames 200 --track
```

**Métricas de postura (PCK / acurácia / recall / falso-alarme / latência).** Primeiro crie o ground-truth, depois avalie:

```bash
python scripts/annotate_gt.py --video data/validation/clip.mp4 --output data/annotations/clip_gt.json
# teclas: n=normal e=ergonômico i=immediate_risk z=critical_zone | SPACE pausa | . passo | q finaliza

python scripts/evaluate_pose.py \
  --annotations data/annotations/clip_gt.json \
  --video data/validation/clip.mp4 \
  --device cuda
```

O `evaluate_pose.py` carrega os thresholds de `config/pose_config.yaml` (use `--config` para apontar outro arquivo); `--pose-model`, `--device` e `--zones` são overrides opcionais.

> O ground-truth gerado pelo `annotate_gt.py` serve às métricas de **classificação** (recall, acurácia, falso-alarme). Os keypoints dele vêm do próprio detector, então o PCK reportado é auto-referencial — um PCK independente exige keypoints rotulados manualmente.

---

## Testes

A lógica determinística (motor REBA e máquina de estados do classificador) é coberta por testes unitários — não exigem GPU nem modelo:

```bash
pip install pytest
python -m pytest tests/ -q
```

Os testes do REBA validam as tabelas A/B/C contra o worksheet canônico de Hignett & McAtamney; os do classificador travam a histerese assimétrica e a detecção de queda (o que protege a meta de recall ≥ 0,90).

---

## Treino (detector da Fase 1)

```bash
python scripts/download_dataset.py          # baixa o dataset Construction-PPE
python src/train.py --model yolo11n.pt --epochs 100 --imgsz 640 --batch 16
cp outputs/runs/ppe_train/weights/best.pt models/exp.pt
```

---

## Documentação adicional

- [docs/taxonomy.md](docs/taxonomy.md) — taxonomia de posturas, REBA e máquina de estados.
- [docs/auditoria-dia1-status.md](docs/auditoria-dia1-status.md) — status dos entregáveis do Dia 1.
- [docs/metricas-dia2.md](docs/metricas-dia2.md) — métricas medidas e calibração do Dia 2.
- [docs/relatorio-dia2.md](docs/relatorio-dia2.md) — relatório consolidado do Dia 2.

---

## Licença

AGPL-3.0 — herdada do dataset Construction-PPE.
