# E2-B — Resultado da verificação de ponta a ponta

Data: 2026-08-20
Branch: `feat/e2b-capture-resiliente` (a partir de `main`, commit `5d0f52b`)

Este documento registra o que foi observado ao executar a Task 7 do plano
`2026-08-20-e2b-capture-resiliente.md`. Ele não é um certificado — o que não
pôde ser verificado está listado como tal, não maquiado.

## O que foi verificado

### Suíte automatizada

```
cd /home/asaas/projects/challenge/cypher-capture
./venv/bin/python -m pytest tests/ -q
```

192 testes passam, zero falhas — 138 da suíte de E1/E1-B (linha de base
antes deste plano) mais 54 novos de E2-B (`test_placeholder_expansion.py`,
`test_camera_config_client.py`, `test_resilient_capture.py`,
`test_capture_config.py`). Nenhum teste pré-existente mudou de
comportamento.

Cobrem, com transporte/captura injetados (sem rede, sem câmera, sem GPU):

- expansão de `${VAR}` e falha nomeada quando falta credencial (Task 1);
- a escada de degradação do `CameraConfigClient` — API, cache, `None` —
  incluindo os dois casos que **não** degradam (409 desativada, 404 não
  cadastrada) e a garantia de que o cache nunca grava a credencial expandida
  (Task 2);
- `is_live_source` para os esquemas de câmera e para arquivo/diretório
  (Task 3);
- `ResilientCapture` em modo arquivo, quadro a quadro, sem descarte (Task 4);
- `ResilientCapture` em modo ao vivo: thread de leitura, descarte do frame
  atrasado, e reconexão com backoff — incluindo um bug real encontrado
  durante a Task 5 (a thread morria para sempre se a primeira tentativa de
  reconexão falhasse) e corrigido antes de seguir, com um teste de regressão
  cobrindo várias falhas consecutivas antes do sucesso (Task 5);
- `CaptureConfig`, `aplicar_opcoes_ffmpeg` (TCP por padrão, as duas grafias
  de timeout, sem sobrescrever o operador) (Task 6).

### Regressão manual contra `cenario2.mp4`

```
./venv/bin/python src/run_pipeline.py --source data/validation/cenario2.mp4 \
  --device cuda --no-display --no-stream --max-frames 120
```

executado pelo implementador da Task 6 (execução completa, 1331/1331
frames — todo o clipe, não só os 120 do comando do plano):

- linha `[config] fonte=data/validation/cenario2.mp4 (origem: --source)`
  confirma o caminho local, sem tocar no cadastro;
- `cap.fps` reportou o fps real do arquivo (29.98), não o fallback fixo de
  `ResilientCapture` (15.0) — importante porque esse fallback é um valor
  fixo, diferente do `config.target_fps` que o código antigo usava; o
  clipe não bate nesse caminho, mas fica registrado como algo a observar se
  algum dia uma fonte não reportar fps;
- latência: PASS, p95 = 57.7ms (orçamento 200ms);
- contagem de frames bate com `cv2.CAP_PROP_FRAME_COUNT` do arquivo — a
  captura por arquivo continua lendo quadro a quadro sem descartar.

Sem `--camera-id`, o comportamento é o de antes deste plano.

## O que não pôde ser verificado nesta sessão

### Passos 1–5 (API do cadastro): E2-A não está implementado no backend

Os passos 1–5 desta tarefa pedem subir a API Spring Boot em
`/home/asaas/projects/cypher`, cadastrar estação e câmeras via
`POST /api/v1/cameras`, buscar `GET /api/v1/cameras/{id}/config`, e
desativar/reativar via `DELETE`/`PUT`. Inspecionando esse repositório:

```
$ grep -rln "RestController" /home/asaas/projects/cypher/src/main/java
.../controller/SummaryController.java
.../controller/CatalogController.java
.../controller/IncidentController.java
```

Só existe `GET /api/v1/cameras` (lista, em `CatalogController`, aparentemente
sobre um catálogo estático/seedado) e `GET /api/v1/stations`. Não existem
`POST`, `PUT`, `DELETE` para câmeras/estações, nem
`GET /api/v1/cameras/{id}/config`. Ou seja, **E2-A ainda não foi
implementado nesse backend** — o plano já reconhece essa dependência
("Depende de: E2-A Tasks 1–7 ... precisa existir para a Task 7 deste
plano"), e as Tasks 1–6 deste plano foram desenhadas justamente para não
precisar dela (o transporte é injetado nos testes).

Como consequência, os passos 1–5 não puderam ser executados contra uma API
real nesta sessão:

- Passo 1–2: não há endpoint para cadastrar a câmera nem para buscar sua
  config — não dá para confirmar `[config] fonte=... (origem: cadastro)`
  contra uma API real.
- Passo 3 (escada com API fora / sem cache): a lógica está coberta pelos
  testes injetados de `test_camera_config_client.py` (API indisponível cai
  pro cache; sem cache, `None`), mas não foi observada contra uma API real
  subindo e caindo.
- Passo 4 (credencial ausente): coberto por `test_placeholder_expansion.py`
  e `test_camera_config_client.py` com ambiente injetado; não observado via
  cadastro real com placeholder.
- Passo 5 (câmera desativada não cai para o cache): coberto por
  `test_camera_desativada_aborta_e_nao_cai_para_o_cache`; não observado
  contra um `DELETE`/`PUT` reais, que não existem no backend hoje.

**Recomendação:** repetir os passos 1–5 desta tarefa quando E2-A estiver
implementado e mergeado no backend. O lado do pipeline (Tasks 1–6 deste
plano) já está pronto e testado para esse cenário — o que falta verificar é
a integração real, não a lógica.

### Passo 6 (reconexão com câmera IP real): não verificado em hardware

Não havia uma câmera IP disponível nesta sessão para desconectar o cabo de
rede e observar a reconexão real. A lógica de reconexão — thread de
leitura, descarte de frame atrasado, backoff, e a retomada após múltiplas
falhas consecutivas de reabertura — está coberta por
`tests/test_resilient_capture.py` com uma câmera fake controlável
(`CapturaAoVivoFake`), mas isso não substitui o comportamento de uma
câmera e rede reais, como o próprio plano já observa.

## Estado ao fim desta verificação

O pipeline (lado `cypher-capture`) está implementado e testado para toda a
escada de degradação de config e para sobreviver a câmera que cai, trava ou
produz mais rápido do que o processamento consome — com compatibilidade
retroativa confirmada (138 testes de E1/E1-B inalterados, regressão manual
de `cenario2.mp4` idêntica ao comportamento anterior). A verificação
ponta a ponta contra uma API e uma câmera reais depende de E2-A, que não
está implementado no backend nesta sessão, e de hardware de câmera IP, que
não estava disponível.
