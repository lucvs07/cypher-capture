# Cypher Capture — Visão Computacional Industrial

Monitoramento de segurança em tempo real para canteiros de obras, construído sobre [Ultralytics YOLO](https://docs.ultralytics.com/).

O projeto tem duas fases que compartilham a mesma base de código:

- **Fase 1 — Detecção de EPIs:** identifica equipamentos de proteção individual (presentes ou ausentes) a partir de webcam, vídeo ou imagens, e opcionalmente envia os frames de violação para o Cloudinary.
- **Fase 2 — Pose e análise comportamental:** um pipeline unificado que roda estimação de pose (YOLO11-pose + ByteTrack) junto com a detecção de EPIs no mesmo laço, calcula ângulos articulares e um escore ergonômico **REBA** completo, classifica a postura em quatro categorias com histerese temporal e sinaliza a entrada/aproximação em zonas críticas demarcadas.
- **Fase 3 — Integração com o backend:** `run_pipeline.py` reporta incidentes ao `cypher` (Spring Boot) via `X-Cypher-Key` por câmera, resolve fonte de vídeo e zonas críticas a partir do cadastro remoto de câmeras (com fallback em cascata para cache local e arquivo), faz hot-reload de zonas sem reiniciar o processo, envia heartbeat periódico para a gamificação (E5) e expõe o frame anotado como stream MJPEG. Um orquestrador (`src/orchestrator/`, `python -m src.orchestrator`) reconcilia o catálogo de câmeras do backend com os processos `run_pipeline.py` em execução no host — sobe, derruba e reinicia um processo por câmera automaticamente.

Referências normativas: **NR-17** (ergonomia), **NR-35** (trabalho em altura) e **REBA** (Hignett & McAtamney, 2000). A taxonomia completa de posturas está em [docs/taxonomy.md](docs/taxonomy.md). A arquitetura dos três repositórios (`cypher-capture` → `cypher` → `cypher-front`) está documentada em [arquitetura.md do cypher-front](https://github.com/Ronaldovels/cypher-front/blob/dev/docs/arquitetura.md).

---

## Estrutura do projeto

```
cypher-capture/
├── config/
│   ├── construction-ppe.yaml     # dataset da Fase 1 (caminhos + nomes das classes)
│   ├── pose_config.yaml          # fonte única de thresholds e parâmetros (pipeline/classifier/thresholds/zones/notify)
│   └── zones/                    # zones.json (gerado pelo anotador de zonas) + zones.example.json
├── data/
│   ├── annotations/              # ground-truth JSON para avaliação
│   └── validation/               # vídeos de validação locais (fora do versionamento)
├── docs/
│   ├── taxonomy.md               # taxonomia de posturas, REBA, máquina de estados
│   ├── auditoria-dia1-status.md  # status do Dia 1
│   ├── metricas-dia2.md          # métricas medidas no Dia 2
│   ├── relatorio-dia2.md         # relatório consolidado do Dia 2
│   └── ...                       # demais relatórios/auditorias de entrega (dia a dia)
├── models/
│   └── exp.pt                    # detector de EPI treinado (7 classes)
├── scripts/
│   ├── download_dataset.py       # baixa o dataset Construction-PPE
│   ├── annotate_zones.py         # desenha polígonos de zona crítica
│   ├── annotate_gt.py            # rotula a postura de referência por frame
│   ├── benchmark_pose.py         # benchmark de latência do modelo de pose
│   └── evaluate_pose.py          # métricas: PCK / acurácia / recall / latência
├── tests/                        # cobertura unitária de REBA, classificador, capture, notify, zones e orchestrator
└── src/
    ├── detect.py                 # entry-point da Fase 1 (apenas EPI)
    ├── run_pipeline.py           # entry-point da Fase 2/3 (EPI + pose + integração com o backend)
    ├── train.py                  # treino do detector de EPI
    ├── pose/                     # estimador, ângulos, REBA, classificador (com gating temporal de queda), temporal
    ├── zones/                    # ZoneManager, ZoneDetector, ZoneConfigClient (cadastro remoto) e ZoneHotReloader
    ├── fusion/                   # motor de decisão + pipeline unificado
    ├── capture/                  # captura resiliente, cadastro remoto de câmera e provisionamento de apiKey
    ├── notify/                   # envio de incidentes ao backend, heartbeat e stream MJPEG
    ├── orchestrator/             # orquestrador de processos: reconcilia o catálogo de câmeras com run_pipeline.py
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
| Detector de EPI | `models/exp-m4-1709-v4.pt` (padrão; `exp.pt` e `exp-s3-1709.pt` têm as mesmas classes) | detect | 7 classes: `GLOVE, Goggles, HELMET, MASK, Person, VEST, boots` |
| Estimador de pose | `yolo11s-pose.pt` | pose | baixado automaticamente; 17 keypoints COCO |

Para um modelo de pose mais rápido ou mais preciso, troque em `config/pose_config.yaml → pipeline.pose_model_path`: `yolo11n-pose.pt` (mais rápido), `yolo11s-pose.pt` (padrão) ou `yolo11m-pose.pt` (mais preciso).

> O modelo de EPI emite apenas classes de **presença** (não há classes `no_*` de violação). EPI ausente é inferido pelo pipeline: um item exigido (pela zona ou pelo checklist da empresa) que fica sem detecção por `epi_confirm_seconds` seguidos numa pessoa rastreada vira violação (`src/pose/epi_absence.py`). Só os 6 EPIs que o modelo detecta existem no contrato com a API (`EpiTipo`: `HELMET, GOGGLES, GLOVE, VEST, BOOTS, MASK`); `tests/test_epi_catalog.py` falha se os pesos de `models/` mudarem de classes. Para inspecionar as classes de quaisquer pesos:
> `python -c "from ultralytics import YOLO; print(YOLO('models/exp-m4-1709-v4.pt').names)"`

---

## Como rodar — Fase 2 (EPI + pose unificados)

Este é o entry-point principal. Ele lê todos os parâmetros de `config/pose_config.yaml`.

> Para subir os 3 projetos do sistema juntos (backend + front + este pipeline, com dado real da empresa `demo`), use `scripts/subir-empresa-demo.sh` (mesmo script, também presente em `cypher` e `cypher-front` — roda a partir de qualquer um dos três): `./scripts/subir-empresa-demo.sh start` / `status` / `logs` / `stop`. Detalhes: [`arquitetura.md`](https://github.com/Ronaldovels/cypher-front/blob/dev/docs/arquitetura.md) do `cypher-front`. As instruções abaixo são para rodar só este pipeline isoladamente.

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

**Gating temporal (janela de confirmação):** quando o único sinal disparado é a velocidade de queda (`fall_velocity`), o classificador (`src/pose/classifier.py`) não reporta `immediate_risk` de imediato — abre uma janela de confirmação de `fall_confirm_seconds` (padrão 1,0 s, convertido em `fall_confirm_frames` a partir de `target_fps`) e só confirma o alarme se, dentro dessa janela, a postura "caída" for corroborada geometricamente (`person_fallen`) ou permanecer sinalizada; se a janela esgota sem confirmação, o episódio é descartado sem gerar incidente. Isso evita falso alarme em agachamentos rápidos, que geram pico de velocidade sem configurar queda de fato. Sinais que já chegam confirmados (ex.: `person_fallen`, `balance_loss_risk`) não passam por essa janela — o gating vale só para `fall_velocity` puro.

Todos os limiares são configuráveis em `config/pose_config.yaml`, sem alterar o código.

---

## Integração com o backend (`cypher`)

Com `--camera-id` e `--api-url` (ou as chaves equivalentes na seção `notify:` de `config/pose_config.yaml`), `run_pipeline.py` deixa de rodar isolado e passa a se comportar como o pipeline de uma câmera cadastrada no backend:

```bash
python src/run_pipeline.py --camera-id CAM-01 --api-url http://localhost:8080/api/v1 --device cuda --no-display
```

- **Fonte de vídeo pelo cadastro:** em vez de `--source`, a URL do stream vem de `GET /cameras/{id}/config`. Se o cadastro estiver indisponível, cai para um cache local (`outputs/`) e, na ausência de ambos, exige `--source` explícito para depuração — `--source` sempre vence o cadastro quando os dois são informados. Placeholders `${VAR}` na URL (ex.: `rtsp://${U}:${P}@host/stream`) são expandidos a partir de variáveis de ambiente do host, para a senha da câmera nunca trafegar pela rede nem ficar no cadastro.
- **Autenticação por câmera (`X-Cypher-Key`):** todo incidente e heartbeat enviado ao backend carrega esse header. A chave vem de uma variável de ambiente referenciada por `notify.api_key_env` (ex.: `CYPHER_KEY_CAM01` no `.env`), **ou** é provisionada automaticamente no primeiro boot — login como conta de serviço `GESTOR` (`CYPHER_GESTOR_USER`/`CYPHER_GESTOR_PASSWORD`) seguido de `POST /cameras/{id}/apikey/regenerate` — e cacheada em `outputs/apikey-cache.json` para não rotacionar de novo a cada execução.
- **Zonas críticas pelo cadastro:** com `--camera-id`/`--api-url`, as zonas também podem vir do backend em vez do JSON local, numa escada de degradação: API responde → converte normalizado→pixel com a resolução real do frame e grava cache; API fora do ar → usa o cache; sem cache → mantém o arquivo local (`config.zones_config`). `--zones` (arquivo explícito) sempre vence o cadastro.
- **Hot-reload de zonas:** por padrão o pipeline rebusca as zonas remotas a cada `--zone-refresh-seconds` (30s) numa thread separada e aplica a mudança sem reiniciar o processo — só reaplica quando a lista de zonas muda de fato. `--zone-refresh-seconds 0` ou `--zones` desativam.
- **Envio de incidentes:** `HttpIncidentSink` enfileira aberturas (`POST /incidents`) e fechamentos (`PATCH /incidents/{key}`) numa fila em memória processada por uma thread dedicada, fora do laço de frames — retry com backoff exponencial, spill em disco (`outputs/incidents.jsonl`) quando a fila enche ou o backend está fora do ar, e reenvio do spill no próximo boot. Sob fila cheia, descarta primeiro o incidente de **menor severidade**, nunca o mais recente. Snapshot do frame é enviado por upload assíncrono após o `POST` (nunca bloqueia a abertura do incidente).
- **Heartbeat (gamificação, E5):** `HeartbeatSender` manda `POST /cameras/{id}/heartbeat` a cada `notify.heartbeat_interval_seconds` (30s por padrão) enquanto o pipeline está de pé, alimentando a métrica de cobertura da câmera no backend. Sem `api_key`/`api_key_env` resolvido, o heartbeat fica desativado (só loga um aviso).
- **Stream de vídeo ao vivo:** um servidor MJPEG (`src/notify/mjpeg.py`) publica o frame anotado em `http://127.0.0.1:<porta>/stream` (porta padrão 8090, ou a definida no cadastro da câmera). Desative com `--no-stream`.

| Flag adicional (`run_pipeline.py`) | Padrão | Descrição |
|---|---|---|
| `--camera-id` | do config | identificador da câmera no cadastro do backend |
| `--api-url` | do config | base da API do `cypher` (ex.: `http://localhost:8080/api/v1`); sem ela os incidentes só são impressos |
| `--stream-port` | do config (8090) | porta do stream MJPEG |
| `--no-stream` | desligado | não sobe o servidor de vídeo ao vivo |
| `--zone-refresh-seconds` | `30` | intervalo do hot-reload de zonas remotas; `0` desativa |
| `--rotate` | `0` | `90`/`180`/`270`: corrige uma câmera montada fisicamente virada, aplicado antes de qualquer análise |

Sem `--api-url`, nada disso entra em jogo: o pipeline se comporta exatamente como na Fase 2 (`ConsoleIncidentSink` imprime os eventos, sem rede).

---

## Orquestrador de processos (`src/orchestrator/`)

Para não subir cada câmera manualmente com `run_pipeline.py`, o orquestrador reconcilia periodicamente o catálogo de câmeras do backend com os processos em execução no host — um `run_pipeline.py --no-display` por câmera:

```bash
python -m src.orchestrator --api-url http://localhost:8080/api/v1 --max-concorrentes 2
```

- **`OrchestratorSession`** loga como conta de serviço `GESTOR` (`CYPHER_GESTOR_USER`/`CYPHER_GESTOR_PASSWORD`) e faz polling em `GET /cameras`; se a sessão expira (401), reloga sozinha antes de propagar falha — qualquer problema de sessão ou rede vira um único caso para quem chama: "catálogo inacessível agora" (o ciclo simplesmente não mexe em nada).
- **`ProcessRegistry`** sabe quais câmeras têm processo vivo e com qual `streamUrl` cada uma subiu; persiste o PID de cada processo em `outputs/orchestrator-state.json` para sobreviver a um restart do próprio orquestrador sem duplicar processo — no boot, mata qualquer PID órfão deixado pela execução anterior.
- **`Reconciler`** a cada ciclo (`--intervalo`, padrão 30s): encerra processos de câmeras que saíram do catálogo, reinicia os que mudaram de `streamUrl` ou morreram sozinhos (com **backoff exponencial** por câmera, mesma fórmula da captura resiliente — primeira falha espera `backoff_base_seconds`, dobra a cada falha subsequente até `backoff_max_seconds`, lidos de `config/pose_config.yaml`), e sobe câmeras novas — tudo respeitando o teto de `--max-concorrentes` pipelines simultâneos no host, com prioridade para reinícios sobre câmeras novas.
- Cada processo filho roda com `--no-display` e tem seu próprio log em `outputs/orchestrator-logs/<camera_id>.log`.

| Flag | Padrão | Descrição |
|---|---|---|
| `--api-url` | — | URL base da API do `cypher` para este tenant (obrigatório) |
| `--intervalo` | `30` | segundos entre ciclos de reconciliação |
| `--max-concorrentes` | `2` | teto de pipelines simultâneos neste host |
| `--config` | `config/pose_config.yaml` | de onde lê `backoff_base_seconds`/`backoff_max_seconds` |
| `--state-path` | `outputs/orchestrator-state.json` | onde persiste o mapa `camera_id -> pid` |
| `--log-dir` | `outputs/orchestrator-logs` | diretório de log por câmera |

O orquestrador é **mono-tenant por instância**: rodar câmeras de mais de uma empresa no mesmo host exige um processo de orquestrador por empresa, cada um com sua própria conta `GESTOR` (decisão de arquitetura registrada em [arquitetura.md](https://github.com/Ronaldovels/cypher-front/blob/dev/docs/arquitetura.md) do `cypher-front`, não uma limitação pendente).

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

A lógica determinística é coberta por testes unitários — não exigem GPU nem modelo, e transportes de rede reais (requests) são substituídos por fakes/protocols em todos eles:

```bash
pip install pytest
python -m pytest tests/ -q
```

Os testes do REBA validam as tabelas A/B/C contra o worksheet canônico de Hignett & McAtamney; os do classificador travam a histerese assimétrica, o gating temporal de queda e a detecção de queda em si (o que protege a meta de recall ≥ 0,90). O restante da suíte cobre `src/capture/` (captura resiliente, cadastro remoto de câmera, provisionamento de apiKey), `src/notify/` (fila de incidentes, heartbeat, stream MJPEG), `src/zones/` (cadastro remoto e hot-reload de zonas) e `src/orchestrator/` (sessão, registro de processos, reconciliação e CLI).

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

## Vídeo de demonstração

https://github.com/user-attachments/assets/da582c0c-9bc6-4ca3-9de6-844e875141b9

Vídeo único, centralizado no repositório principal ([`cypher-front`](https://github.com/Ronaldovels/cypher-front#-vídeo-de-demonstração)), navegando pelas principais telas do sistema com os 3 projetos rodando localmente (dado real da empresa `demo`).

### Vídeos do pitch final (Fase 4)

Também centralizados no [`cypher-front`](https://github.com/Ronaldovels/cypher-front/tree/main/docs/apresentacao/videos), gravados com o sistema rodando — o feed ao vivo é o `run_pipeline.py` rodando sobre os vídeos de `data/validation/` (girados para 1920x1080) no lugar das câmeras RTSP:

- [Demonstração ao vivo — 60s](https://github.com/Ronaldovels/cypher-front/blob/main/docs/apresentacao/videos/pitch-final-demo-ao-vivo.mp4): backup do slide 12, com detecção → decisão → alerta nas duas telas → análise por IA.
- [Tour pelas telas — 1min45](https://github.com/Ronaldovels/cypher-front/blob/main/docs/apresentacao/videos/pitch-final-tour-telas.mp4): login, dashboard, ao vivo, heatmap, incidentes, placar, EPIs, zonas de risco e MCP.
- [Completo — 2min45](https://github.com/Ronaldovels/cypher-front/blob/main/docs/apresentacao/videos/pitch-final-completo.mp4): tour seguido da demonstração ao vivo.

## Gestão ágil (Sprint 3)

- Board Trello: https://trello.com/b/AqgHhHCP/dev4-cypher-challenge
- Protótipo Figma: https://www.figma.com/design/kyeXTO2Uw0lkm8DPK2a22v/Interface?node-id=48-2
- Design system: https://github.com/Ronaldovels/cypher-front/blob/dev/docs/design-system.md
- Arquitetura (3 repositórios): https://github.com/Ronaldovels/cypher-front/blob/dev/docs/arquitetura.md
- Fluxo de telas: https://github.com/Ronaldovels/cypher-front/blob/dev/docs/fluxo-de-telas.md
- Repositório principal da entrega: https://github.com/Ronaldovels/cypher-front

---

## Licença

AGPL-3.0 — herdada do dataset Construction-PPE.
