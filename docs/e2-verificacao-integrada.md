# E2 — Verificação integrada (cypher + cypher-capture + cypher-front)

Executado em 2026-08-23, com os três repositórios rodando simultaneamente nas branches
dos PRs abertos:

- `cypher` (Cypher-Api): worktree `/home/asaas/projects/cypher/.claude/worktrees/feat+e2a-backend-cadastro`,
  branch `feat/e2a-backend-cadastro`, PR [#3](https://github.com/lucvs07/Cypher-Api/pull/3).
- `cypher-capture`: checkout principal, branch `feat/e2b-capture-resiliente`,
  PR [#6](https://github.com/lucvs07/cypher-capture/pull/6).
- `cypher-front`: worktree `.worktrees/e2c-front-cadastro`, branch `e2c-front-cadastro`,
  PR [#2](https://github.com/Ronaldovels/cypher-front/pull/2).

Câmera usada: webcam real do notebook (`/dev/video0`), inferência em `cuda`
(`nvidia-smi` confirmou GPU disponível). Backend: `./mvnw -o spring-boot:run` (H2 em
arquivo, `ddl-auto: update` — persistência real, não em memória). Front: `ng serve`.

Este documento cobre exatamente o que a suíte de 201 testes de E2-B **não** pode
provar por desenho (transporte e captura fake): resposta HTTP real, `cv2.VideoCapture`
real, e a cadeia completa com o front.

## Suítes automatizadas nas três branches (pré-condição)

| Repo | Comando | Resultado |
|---|---|---|
| cypher (E2-A) | `./mvnw -o test` | **134/134** passed, BUILD SUCCESS |
| cypher-capture (E2-B) | `./venv/bin/python -m pytest -q` | **201/201** passed — bate com o que o PR #6 afirma |
| cypher-front (E2-C) | `npx ng test --no-watch` | **115/115** passed (15 arquivos) |

## Contrato do backend (pré-condição de tudo abaixo)

`CatalogController` no worktree `feat/e2a-backend-cadastro` tem `POST`/`PUT`/`DELETE`
em `/stations` e `/cameras`, mais `GET /cameras/{id}/config` — confirmado por leitura
de código antes de subir qualquer serviço.

Evidência real (curl contra o backend rodando, não paráfrase):

```
POST /stations {"id":"EST-30",...}                → 201 {"id":"EST-30",...,"ativa":true,"cameras":[]}
POST /stations {"id":"EST-30",...} (duplicado)     → 409 {"erro":"Estacao ja cadastrada: EST-30"}
POST /cameras estacaoId=EST-99 (inexistente)       → 422 {"erro":"Estacao inexistente: EST-99"}
PUT /stations/EST-30 {"id":"EST-31",...}           → 409 {"erro":"O codigo de estacao e imutavel: EST-30 nao pode virar EST-31"}
POST /cameras streamUrl duplicada de CAM-30        → 409 {"erro":"streamUrl ja usada pela camera CAM-30"}
GET /cameras (lista)                               → sem campo "fonte"
GET /cameras/CAM-01 (item)                         → com campo "fonte"
GET /stations (sem incluirInativas)                → só ativas
GET /cameras?incluirInativas=true                   → inclui inativas
OPTIONS preflight (Origin localhost:4200)          → Access-Control-Allow-Credentials: true
```

Restart do backend com EST-30/CAM-30 já cadastradas e 2 incidentes de teste
ingeridos: após `pkill` + novo `spring-boot:run`, `GET /stations?incluirInativas=true`
e `GET /incidents` devolveram os mesmos dados — seeder é bootstrap-only, não
sobrescreve edição (confirma E2-A §1 e E2-C Step 6).

### Pergunta em aberto respondida com evidência: cascata de desativação de estação

Os dois prompts de validação (E2-A §2 e E2-C §5) fazem a mesma pergunta em termos
diferentes e ambos ficam certos ao mesmo tempo:

- `DELETE /stations/EST-30` **não** desativa `CAM-30` no banco — `GET /cameras`
  continua listando CAM-30 como ativa depois. O mock da E2-C acertou a suposição.
- Mas `GET /cameras/CAM-30/config` (a superfície que o pipeline usa) devolve
  **409** `"Estacao da camera esta desativada: EST-30"` mesmo com `CAM-30.ativa=true`
  — então o pipeline não sobe apontando para uma câmera cuja estação foi
  desativada, apesar de a câmera em si continuar "ativa" para o front.

Reativando a estação, `/config` volta a 200 imediatamente — confirmado.

### Ingestão permissiva confirmada

`POST /incidents` com `cameraId` desconhecida → **201** (aceito). Com câmera já
desativada → também **201**. Cadastro é estrito, ingestão nunca perde dado — exatamente
como o docstring do `CatalogService` promete.

## Cenários A–I (pipeline real contra webcam e backend reais)

### A. Cadastro feliz — fonte vem da API, não de `--source` ✅

```
[config] fonte=0 (origem: cadastro)
[stream] video ao vivo em http://127.0.0.1:8130/stream
Latência: p95=39.1ms, PASS (≤200ms) — 150 frames, cuda, webcam real
```
`streamPort` (8130) veio do cadastro, não do YAML (que é 8090). MJPEG server
efetivamente sobe na porta derivada. `outputs/camera-config-cache.json` gravado
com o mesmo conteúdo do cadastro.

### B. Escada de degradação com a API real caindo ✅

Com backend derrubado (`pkill`) e cache já gravado:
```
Config de CAM-30 indisponivel na API: ...Connection refused...
[config] fonte=0 (origem: cache de 0.0h)
```
Com cache apagado e API ainda fora:
```
[config] cadastro indisponivel e sem cache para CAM-30; informe --source
Informe --source, ou --camera-id com --api-url apontando para uma camera cadastrada.
```
Sem stack trace nos dois casos, processo termina limpo (exit 0).

### C. `--source` vence `--camera-id` mesmo com erro definitivo de cadastro ✅ (bug crítico corrigido, confirmado)

Câmera desativada via API (`DELETE /cameras/CAM-30` → 409 real no `/config`),
pipeline chamado com `--camera-id CAM-30 --api-url ... --source 0`:
```
[config] cadastro de CAM-30 indisponivel (Camera desativada: CAM-30); seguindo com --source
[config] fonte=0 (origem: --source)
```
Não crasha, segue com o arquivo/webcam local. Confirmado.

**Achado a registrar (não testado experimentalmente, confirmado por leitura de
código em `src/run_pipeline.py:155-162`):** quando cai nesse caminho, `stream_port`
só é herdado do cadastro se `not args.source` — ou seja, com `--source` presente
(explícito ou por fallback), a porta do stream cai para o default do YAML (8090
nesta instalação), **que é exatamente a porta registrada para CAM-01**. A garantia
de unicidade de `streamUrl` que o backend aplica em `exigirUrlLivre()` só vale no
cadastro; esse caminho de fallback não a consulta. Se um pipeline de CAM-01
orientado por cadastro estiver rodando ao mesmo tempo, os dois vão tentar bind na
mesma porta. Não reproduzido nesta sessão (exigiria dois pipelines simultâneos);
recomendo logar um aviso quando o fallback usa a porta default sem `--stream-port`
explícito.

*Nota de medição:* o relatório final desta execução apontou `Latência: FAIL
(p95=983.3ms)`, mas é artefato de amostra pequena (15 frames, primeiro frame frio
em 3.1s puxando o p95) — não é regressão; o mesmo pipeline com warmup (cenário A,
150 frames) passa com folga.

### D. Câmera desativada não cai para o cache — só o caminho 409 pôde ser exercitado ⚠️ parcial

Com cache válido em disco (`ativa: true`) e câmera desativada via `DELETE`:
```
[config] Camera desativada: CAM-30
```
Aborta sem usar o cache antigo — confirmado.

**Não pôde ser testado contra este backend:** o segundo caminho que o item D pede
— um `PUT`/`PATCH` devolvendo **200** com `ativa=false` — é estruturalmente
inalcançável aqui. `CatalogService.cameraConfig()` sempre lança `409` quando
`!camera.isAtiva()`; nunca devolve 200 com o corpo carregando `ativa=false`. A
checagem defensiva do pipeline (`CameraConfig.ativa` não depender só do status
HTTP) existe no código mas está sem cobertura de integração real contra este
backend — é proteção prospectiva, não algo hoje exercitável de ponta a ponta.

### E. Câmera não cadastrada ✅

```
[config] Camera nao cadastrada: CAM-NAO-EXISTE. Cadastre-a ou use --source.
```
Erro claro, sem stack trace, exit limpo.

### F. Reconexão contra câmera real/RTSP simulado — não executado ❌

Não há câmera IP na rede deste ambiente, e nenhum servidor RTSP (mediamtx ou
similar) foi instalado para simular a queda de rede. Não construí essa infra ad
hoc para não introduzir uma variável não pedida pelo escopo desta sessão. Fica
como pendência explícita, aceita pelos dois documentos-fonte quando falta
hardware.

### G. Frontend reflete o que o pipeline envia — confirmado por partes, não simultaneamente ⚠️ parcial

`/dev/video0` só admite um dono por vez, então o pipeline (webcam) e a sessão de
browser real (Playwright/Chromium, ver seção do front abaixo) não puderam rodar
ao mesmo tempo nesta máquina. O que foi confirmado:
- Incidentes postados via pipeline real (`POST /incidents` a partir de um run de
  pipeline de verdade) aparecem em `GET /incidents` e sobrevivem a restart do
  backend.
- O front lê exatamente esses mesmos endpoints (`CatalogService`/`IncidentService`
  usam `${environment.apiUrl}/cameras`, `/stations`, `/incidents` — mesma origem
  testada aqui).
- Estação/câmera cadastradas pela UI real aparecem no heatmap via navegação SPA
  sem reload (ver relatório do front).

O que **não** foi observado nesta sessão: um pipeline rodando ao vivo, com o
browser aberto ao mesmo tempo, mostrando o stream MJPEG e um incidente chegando
em tempo real via SSE no dashboard. Recomendo repetir com duas câmeras (uma para
o pipeline, outra webcam ou arquivo simultâneo) se esse caminho específico
precisar de certificação visual síncrona.

### H. Compatibilidade retroativa ✅

Backend **derrubado**, pipeline chamado só com `--source data/validation/cenario2.mp4`
(sem `--camera-id`, sem `--api-url`):
```
Frames         : 100
Wall clock     : 8.5s  (11.7 FPS effective)
E2E latência   : p95=40.4ms
Latência target: PASS ✓
```
Roda inteiro sem tentar contato com a API (nenhuma linha `[notify]`, nenhum erro
de conexão) — confirma que backend/front não são pré-requisito desse caminho.

### I. Câmera cai e não volta, com `--max-frames` definido — não executado ❌

Não simulado nesta sessão por custo de tempo/infra (exigiria forçar perda
indefinida de sinal da webcam de forma controlada). Aceito como limitação
conhecida já registrada na revisão final de E2-B; não é bloqueante.

## Idempotência do pipeline (não-funcional)

Confirmado por leitura de código (`grep` em todo `cypher-capture/src`): o pipeline
nunca emite `POST`/`PUT` para `/stations` ou `/cameras` — só lê `/config` (GET) e
escreve incidentes com `incidentKey` novo por run (`run_id` é um UUID por
execução). Reiniciar o pipeline não pode duplicar cadastro porque ele nunca cria
cadastro; e o `IncidentController` documenta idempotência por `incidentKey` do
lado do backend.

## Resumo

| Cenário | Resultado |
|---|---|
| Contrato de escrita do backend (E2-A) | ✅ Confirmado integralmente com evidência real |
| Cascata de desativação (estação → câmera) | ✅ Respondida: sem cascata no banco, mas `/config` bloqueia (409) |
| Restart do backend preserva dados | ✅ Confirmado |
| Ingestão permissiva (câmera desconhecida/desativada) | ✅ Confirmado |
| A — fonte/porta vêm do cadastro | ✅ Confirmado |
| B — escada de degradação (cache → erro claro) | ✅ Confirmado |
| C — `--source` vence cadastro com erro definitivo | ✅ Confirmado (bug crítico corrigido) — porta de fallback é um achado novo, ver acima |
| D — câmera desativada não usa cache velho | ⚠️ Só o caminho 409 é alcançável neste backend; caminho 200/`ativa=false` é código morto para fins de integração |
| E — câmera não cadastrada | ✅ Confirmado |
| F — reconexão de câmera IP real | ❌ Não executado — falta hardware/infra |
| G — front reflete pipeline em tempo real | ⚠️ Confirmado por partes; não simultaneamente (limite de 1 webcam) |
| H — compatibilidade retroativa | ✅ Confirmado |
| I — reconexão indefinida vs `--max-frames` | ❌ Não executado — limitação conhecida, aceita |
| Suíte automatizada (201 testes) | ✅ Confirmado na branch do PR #6 |

Este documento registra o que foi observado; não certifica os itens marcados ❌/⚠️
como concluídos.
