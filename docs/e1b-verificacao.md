# E1-B — verificação de ponta a ponta

Resultado observado ao rodar o pipeline contra a API de incidentes (E1-A,
branch `feat/e1a-backend-incidentes`, H2 em arquivo). **Os números abaixo são
os medidos, não os esperados pelo plano** — onde divergem, a divergência está
explicada.

Ambiente: `cenario2.mp4` (1331 frames, 30 fps, 2 quedas reais), `--device cuda`,
API em `http://localhost:8080/api/v1`.

## 1. Execução completa contra a API (CAM-03, versão final)

| Medida | Valor |
|---|---|
| Frames processados | 1331 |
| Latência E2E | mean 34,2 ms · p50 33,2 ms · **p95 34,3 ms** · max 1069 ms |
| Veredito do orçamento | **PASS** (p95 ≤ 200 ms) |
| Incidentes na API | 3 |
| Incidentes sem `endedAt` | **0** |
| `closedReason = PIPELINE_STOPPED` | 1 |
| Linhas em `outputs/incidents.jsonl` ao final | **0** |

Os três incidentes:

| Tipo | sev | pico | duração | fechamento | reason |
|---|---|---|---|---|---|
| FALL_RISK | 4 | 4 | 17,06 s | ESCALATED | fall_velocity |
| ERGONOMIC_RISK | 2 | 3 | 3,72 s | STATE_EXIT | prolonged_squat |
| FALL_RISK | 4 | 4 | 30,10 s | PIPELINE_STOPPED | fall_velocity |

SSE (`curl -N .../incidents/stream`): **3 eventos `incident`** (um por abertura)
e **9 eventos `incident-changed`** (fechamentos, incluindo os re-fechamentos
que a janela de merge produz ao reabrir um episódio).

Vídeo ao vivo: verificado por `curl`, não por browser. `GET /stream` na porta
8090 devolve `HTTP 200`, `Content-Type: multipart/x-mixed-replace;
boundary=cypherframe`, e quadros JPEG reais de ~172 KB com cabeçalho JFIF.

### Divergência: 3 incidentes, não 2

O plano previa `totalElements == 2`. A execução real produz **3**, e a diferença
não é defeito:

* São **2 FALL_RISK — um por queda real** — mais 1 ERGONOMIC_RISK do agachamento
  prolongado entre elas. Pelo critério "um incidente por queda", 3 é o resultado
  correto; o esperado 2 vinha do replay da fixture, onde as duas quedas caíam
  **dentro** da janela de merge e viravam um episódio só.
* No replay da fixture o intervalo entre o fim do primeiro episódio de queda e a
  reentrada em alarme é de 11,6 s (< 15 s → merge). Na execução real o mesmo
  intervalo é de ~17,9 s (> 15 s → incidente novo).
* A causa do intervalo maior: **o tracker mede a janela de merge em tempo de
  relógio, não em tempo de vídeo.** O clipe tem 44,4 s de duração nominal
  (1331 ÷ 30 fps), mas o pipeline o processa a ~28 FPS efetivos, esticando os
  intervalos. Para câmera ao vivo os dois tempos coincidem; para arquivo, não.
  Registrado como observação, não corrigido: o alvo de produção é câmera ao vivo.

## 2. Comportamento com a API fora do ar

Pipeline rodado com a API derrubada (`--max-frames 400`):

* O pipeline **não parou** — 400 frames processados, saída normal.
* Latência **não degradou**: p95 34,4 ms. O retry e o backoff acontecem na
  thread do sink; a thread do frame nunca espera pela rede.
* `outputs/incidents.jsonl` passou de 1 para **8 linhas** — nada se perdeu.

Com a API de volta e o pipeline rodado com `--max-frames 1`:

* Spill drenado na subida: **8 → 0 linhas**.
* Contagem na API: **3 → 5 incidentes**, com os fechamentos pendentes aplicados.

## 3. Defeito encontrado e corrigido durante a verificação

A primeira execução completa (CAM-02) terminou com o fechamento final gravado
em disco em vez de entregue, e a API mostrava `STATE_EXIT` onde deveria mostrar
`PIPELINE_STOPPED`. Causa: `run_pipeline` chama `tracker.flush()` e
`sink.stop()` em sequência dentro do `finally`; os PATCH de fechamento entravam
na fila microssegundos antes de `stop()` marcar a parada, e o worker morria sem
entregá-los.

Corrigido em `HttpIncidentSink.stop()`, que agora drena a fila (e espera o item
em voo) antes de sinalizar a parada — commit `fix(notify): drena a fila antes de
parar o sink`. Coberto por `test_parada_entrega_o_que_ainda_esta_na_fila`, que
falha sem a correção.

Depois da correção, a execução limpa da seção 1 terminou com o spill **vazio** e
o `PIPELINE_STOPPED` entregue diretamente, sem passar pelo disco.

## 4. Compatibilidade para trás

Sem `--api-url`, o pipeline se comporta como antes: imprime
`[notify] sem api_url: incidentes apenas impressos`, mantém o HUD de latência e
reporta p95 34,2 ms contra o orçamento de 200 ms. A suíte de testes tem **138
testes verdes** (89 eram a linha de base antes deste plano).
