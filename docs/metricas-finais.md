# Métricas finais do modelo e do pipeline

Cada número deste documento aponta para o arquivo que o gerou. Nada foi arredondado nem estimado: onde a fonte traz três casas, o texto traz três casas.

**Fontes:**
- Logs de execução em [`evidencias-sprint4/logs/`](evidencias-sprint4/logs/), gerados com `scripts/evaluate_pose.py` e `pytest`. Hardware e versões em [`ambiente.txt`](evidencias-sprint4/logs/ambiente.txt).
- Relatório final da Fase 4 (`relatorio-final-cypher-fase4-v2`, seções 2.2 a 2.5), para o treino do detector de EPI na Ultralytics Platform. Esses números vêm da plataforma e não de um log deste repositório; a coluna "Fonte" diz de onde cada um saiu.

## 1. Detector de EPI (`models/exp-m4-1709-v4.pt`)

Modelo em uso: YOLO26m, 7 classes (`GLOVE, Goggles, HELMET, MASK, Person, VEST, boots`), obtido por fine-tuning a partir do run `exp-m4-1709`. Treino: 14 experimentos, 640 × 640, até 100 épocas com early stopping (patience 25), em NVIDIA RTX PRO 6000 na Ultralytics Platform.

| Métrica (split de validação) | Valor | Fonte |
|---|---:|---|
| mAP@0.5 | 0,915 | Relatório Fase 4, §2.3 e §2.4 |
| mAP@0.5:0.95 | 0,684 | Relatório Fase 4, §2.3 e §2.4 |
| Precisão / recall globais (plataforma) | 0,869 / 0,890 | Relatório Fase 4, §2.4 |
| Macro F1 no limiar de operação (conf. 0,25, IoU 0,45) | 0,877 | Relatório Fase 4, §2.3 |
| Falsos positivos / falsos negativos no limiar de operação | 280 / 739 | Relatório Fase 4, §2.3 |

### Por classe (limiar de operação: confiança 0,25, IoU 0,45)

Derivado da matriz de confusão de validação. Fonte: Relatório Fase 4, §2.4.

| Classe | Precisão | Recall | F1 |
|---|---:|---:|---:|
| Colete | 0,988 | 0,936 | 0,961 |
| Capacete | 0,971 | 0,931 | 0,951 |
| Pessoa | 0,975 | 0,909 | 0,941 |
| Bota | 0,956 | 0,890 | 0,922 |
| Luva | 0,925 | 0,880 | 0,902 |
| Máscara | 0,723 | 0,797 | 0,758 |
| Óculos | 0,860 | 0,595 | 0,704 |
| **Macro** | **0,914** | **0,848** | **0,877** |

### Por que `exp-m4-1709-v4`

Fonte: Relatório Fase 4, §2.3. Os três candidatos foram comparados nos mesmos 15 frames de `cenario.mp4`, confiança 0,5.

| Candidato | mAP@0.5 | mAP@0.5:0.95 | Detecções em cena |
|---|---:|---:|---:|
| YOLO26s (`exp-s3-1709`) | 0,897 | 0,648 | 7 |
| YOLOv8m (`exp.pt`, anterior) | 0,807 | 0,526 | 29 |
| **YOLO26m (`exp-m4-1709-v4`)** | **0,915** | **0,684** | **30** |

O YOLO26s tinha mAP maior que o YOLOv8m, mas achou 1 capacete onde os outros dois acharam 11 e 14. Entre dois YOLO26m com mAP quase igual (0,917 e 0,915), o `v4` teve 74% menos falsos positivos no limiar de operação (280 contra 1.083). Em alerta, o falso positivo é o que faz o operador ignorar o sistema.

## 2. Pose, postura e queda (CPU)

Rodadas de `scripts/evaluate_pose.py` com `--device cpu` em Intel Xeon @ 2.10GHz (2 vCPUs), Python 3.11.15, torch 2.14.0+cu130, ultralytics 8.4.163, opencv 4.13.0 ([`ambiente.txt`](evidencias-sprint4/logs/ambiente.txt)). O modelo de pose é `yolo11s-pose.pt`.

| Cenário | Frames | Acurácia geral | Latência média / p95 (ms) | Alarmes falsos | Log |
|---|---:|---:|---:|---|---|
| `cenario` | 1008 | 0,787 | 257,6 / 524,5 | 2 episódios em 578 frames normais | [`eval-cenario.txt`](evidencias-sprint4/logs/eval-cenario.txt) |
| `cenario2` | 1331 | 0,675 | 251,9 / 526,5 | 2 episódios em 678 frames normais | [`eval-cenario2.txt`](evidencias-sprint4/logs/eval-cenario2.txt) |
| `test_bus` | 75 | 1,000 | 293,6 / 631,2 | 0 em 75 frames normais | [`eval-test_bus.txt`](evidencias-sprint4/logs/eval-test_bus.txt) |

PCK@0.1 foi 1.000 nos três cenários (meta ≥ 0,80). A meta de acurácia geral (≥ 0,85) **não** foi atingida em `cenario` nem em `cenario2`.

Precisão, recall e F1 por categoria, nos dois cenários com mais de uma categoria:

| Categoria | `cenario` P / R / F1 | `cenario2` P / R / F1 |
|---|---|---|
| normal | 0,794 / 0,848 / 0,820 | 0,652 / 0,848 / 0,737 |
| ergonomically_inadequate | 0,775 / 0,705 / 0,738 | 1,000 / 0,161 / 0,277 |
| immediate_risk (queda) | 0,000 / 0,000 / 0,000 | 0,678 / 0,915 / 0,779 |
| critical_zone | 0,000 / 0,000 / 0,000 | 0,000 / 0,000 / 0,000 |

`cenario` não tem queda anotada e `critical_zone` não tem anotação em nenhum dos cenários; os zeros dessas linhas são ausência de exemplos, não desempenho. O recall de risco imediato de 0,996 do Relatório Fase 4 (§3.5) foi medido em outra avaliação (dois cenários filmados de frente, configuração final com cues de queda, `exit_frames` 45 e `fall_risk_min` 0,85) e **não** sai destes logs. O `cenario2` acima, com 0,915, usa os logs versionados aqui.

### Cenário ponta a ponta (EPI + pose + incidentes)

[`fluxo-cenario4.txt`](evidencias-sprint4/logs/fluxo-cenario4.txt), 1548 frames, 2027 pessoa-frames:

- 1240 pessoa-frames com EPI presente e 368 com EPI ausente (368 escalados).
- Severidade por pessoa-frame: 991 em sev. 0, 252 em sev. 2, 344 em sev. 3, 440 em sev. 4.
- 2 incidentes de EPI. O resultado em CPU é idêntico ao do golden em CUDA (`summary_cpu` e `summary_golden_cuda` iguais).
- Incidentes abertos e fechados na ordem: ergonomia (sev. 2), EPI (sev. 3, luva/capacete/colete), queda (sev. 4), EPI (sev. 4, capacete/colete).

## 3. Latência e FPS

| Cenário | Device | Latência | Fonte |
|---|---|---|---|
| Avaliação de pose | CPU (Xeon 2.10GHz, 2 vCPUs) | média 251,9 a 293,6 ms; p95 524,5 a 631,2 ms; **falha** a meta de 200 ms | logs `eval-*.txt` acima |
| Pipeline ponta a ponta `cenario4` | CPU | média 531,1 ms; p50 528,9 ms; p95 560,2 ms | [`fluxo-cenario4.txt`](evidencias-sprint4/logs/fluxo-cenario4.txt) |
| Visão, GPU | GPU comercial | p95 44,9 ms; FPS sustentado ~28 | Relatório Fase 4, §2.5 (medido com o detector anterior, YOLOv8m) |

Em CPU o pipeline roda em poucos FPS e não cumpre o orçamento de 200 ms. Ele funciona, e é o que o avaliador usa sem GPU, mas o desempenho de produção exige GPU. As medidas em GPU foram feitas com o detector anterior, que exige 13% mais compute que o atual (79,1 contra 68,4 GFLOPs), então servem como teto conservador.

## 4. Testes automatizados

[`pytest-cypher-capture.txt`](evidencias-sprint4/logs/pytest-cypher-capture.txt): **449 passed, 2 skipped** em 7,32 s. Os 2 pulados são o golden de EPI, opt-in com `CYPHER_GOLDEN=1` ([`pytest-skips.txt`](evidencias-sprint4/logs/pytest-skips.txt)).

## 5. Dataset

Fonte: Relatório Fase 4, §2.1 (Ultralytics Platform, projeto `ppe-cypher`). Imagens de instalações reais são confidenciais e não ficam neste repositório.

| Item | Valor |
|---|---|
| Dataset de treino | `cypherdatasetppe`, versão 4 |
| Base de terceiros | clone do dataset público `ppe-yolov8` |
| Dados produzidos pela equipe | 348 imagens adicionadas entre a v1 (8.126) e a v4 (8.474), cerca de 4,1% do total |
| Total | 8.474 imagens, 8.445 rotuladas, 38.618 instâncias |
| Divisão | 6.779 treino, 1.441 validação, 254 teste (80 / 17 / 3%) |

| Classe | Instâncias | Imagens com a classe |
|---|---:|---:|
| Pessoa (`Person`) | 13.117 | 6.527 |
| Capacete (`HELMET`) | 8.635 | 4.328 |
| Luva (`GLOVE`) | 7.115 | 3.684 |
| Bota (`boots`) | 5.163 | 2.477 |
| Colete (`VEST`) | 2.897 | 2.155 |
| Óculos (`Goggles`) | 1.279 | 1.029 |
| Máscara (`MASK`) | 412 | 341 |

A classe Máscara tem só 412 instâncias, o que combina com o F1 de 0,758 dela na seção 1.

## 6. Limitações conhecidas

- Acurácia geral abaixo da meta em `cenario` (0,787) e `cenario2` (0,675).
- Ergonomia em câmera de perfil: recall de 0,161 em `cenario2`.
- Alarmes falsos nos dois cenários maiores: 2 episódios em cada.
- Óculos (F1 0,704) e máscara (F1 0,758) são as classes mais fracas do detector de EPI.
- Zona crítica (`critical_zone`) não tem anotação nos cenários de avaliação.
- Latência em CPU fora do orçamento (ver §3).
