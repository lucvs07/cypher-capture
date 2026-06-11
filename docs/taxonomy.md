# Taxonomia de Posturas — Cypher Industrial Vision AI
## Fase 2 — Pose Estimation e Análise Comportamental

**Versão:** 1.0  
**Referências normativas:** NR-17 (Ergonomia), NR-35 (Trabalho em Altura), REBA (Hignett & McAtamney, 2000)

---

## 1. Modelo de Keypoints

O sistema utiliza os **17 keypoints COCO** extraídos pelo YOLO11s-pose:

| ID | Nome            | ID | Nome            |
|----|-----------------|----|-----------------| 
| 0  | nose            | 9  | left_wrist      |
| 1  | left_eye        | 10 | right_wrist     |
| 2  | right_eye       | 11 | left_hip        |
| 3  | left_ear        | 12 | right_hip       |
| 4  | right_ear       | 13 | left_knee       |
| 5  | left_shoulder   | 14 | right_knee      |
| 6  | right_shoulder  | 15 | left_ankle      |
| 7  | left_elbow      | 16 | right_ankle     |
| 8  | right_elbow     |    |                 |

**Política de confiança:** keypoints com score < 0,5 são descartados (`None`). Ângulos que dependam de um keypoint indeterminado não são calculados — nunca interpolados ou estimados.

---

## 2. Ângulos Articulares Calculados

| Ângulo | Keypoints | Vértice | Significado ergonômico |
|--------|-----------|---------|------------------------|
| **Flexão de tronco** | mid_shoulder, mid_hip | — (vetor vs. vertical) | Ângulo do vetor quadril→ombro em relação à vertical da imagem. 0° = ereto (em pé ou sentado), 90° = tronco horizontal. Assume câmera sem rotação (roll ≈ 0) |
| **Flexão de pescoço** | nose, mid_shoulder, mid_hip | mid_shoulder | Inclinação da cabeça em relação ao eixo da coluna |
| **Elevação de ombro** (esq/dir) | hip, shoulder, elbow | shoulder | Ângulo interno hip→shoulder→elbow. 180° = braço ao longo do corpo, 90° = braço horizontal |
| **Flexão de cotovelo** (esq/dir) | shoulder, elbow, wrist | elbow | Ângulo interno. 180° = braço estendido, 90° = cotovelo dobrado |
| **Flexão de joelho** (esq/dir) | hip, knee, ankle | knee | Ângulo interno. 180° = perna reta, 90° = agachamento profundo |
| **Rotação de tronco (proxy)** | mid_shoulder, mid_hip | — | Offset horizontal normalizado entre ombros e quadris ÷ altura da bbox |

**Limitação declarada:** visão 2D monocular não captura rotações axiais fora do plano da câmera. O proxy de rotação de tronco é uma aproximação válida apenas para câmera frontal ou lateral.

---

## 3. Score REBA

O motor REBA calcula um score de 1 a 15 via três tabelas de lookup (Hignett & McAtamney, 2000):

### Grupo A — Tronco, Pescoço, Pernas

| Segmento | Faixa angular | Score | Nota |
|----------|---------------|-------|------|
| **Tronco** | flexão ≤5° | 1 | Ereto |
| | 5°–20° | 2 | Leve inclinação |
| | 20°–60° | 3 | Inclinação moderada |
| | >60° | 4 | Inclinação severa |
| **Pescoço** | desvio ≤20° | 1 | Posição neutra |
| | desvio >20° | 2 | Inclinação excessiva |
| **Pernas** | flexão de joelho ≤30° | 1 | Em pé bilateral |
| | 30°–60° | 2 | Flexão moderada |
| | >60° | 3 | Agachamento |

### Grupo B — Braço Superior, Braço Inferior, Punho

| Segmento | Faixa angular (desvio da vertical) | Score |
|----------|------------------------------------|-------|
| **Braço superior** | ≤20° | 1 |
| | 20°–45° | 2 |
| | 45°–90° | 3 |
| | 90°–110° | 4 |
| | >110° | 5 |
| **Braço inferior** | cotovelo 60°–100° | 1 |
| | fora dessa faixa | 2 |
| **Punho** | fixo em 1 | 1 (2D) |

> **Limitação declarada:** desvio e torção de punho não são detectáveis com confiança em 2D. Score fixo em 1 é uma suposição conservadora (subestima o risco real caso haja desvio de punho).

### Tabela C e Nível de Ação

| Score REBA | Nível de Ação | Descrição |
|------------|---------------|-----------|
| 1 | 1 | Risco desprezível — sem ação necessária |
| 2–3 | 2 | Risco baixo — mudança pode ser necessária |
| 4–7 | 3 | Risco médio — investigar e alterar em breve |
| 8–15 | 4 | Risco alto — investigar e alterar imediatamente |

---

## 4. Taxonomia de Categorias de Postura

### Categoria 1 — NORMAL
**Definição:** nenhuma das regras abaixo está ativa.  
**Ação:** nenhuma.

---

### Categoria 2 — ERGONOMICAMENTE INADEQUADA
**Definição:** postura que excede limiares de risco ergonômico segundo NR-17, sustentada por tempo suficiente (histerese temporal).

#### Regra 2.1 — Carga com coluna fletida (`trunk_flexion_spine_load`)
- **Condição:** `trunk_flexion > 60°` **E** `max(knee_flexion_L, knee_flexion_R) > 150°`
- **Justificativa:** levantar ou carregar carga com flexão anterior de tronco acima de 60° e pernas predominantemente estendidas sobrecarga os discos lombares (NR-17 §17.6.2)
- **Keypoints:** mid_shoulder, mid_hip, mid_knee, left_knee, right_knee

#### Regra 2.2 — Agachamento prolongado (`prolonged_squat`)
- **Condição:** `min(knee_flexion_L, knee_flexion_R) < 90°`
- **Justificativa:** agachamento profundo sustentado pressiona cartilagem do joelho e compromete estabilidade postural (NR-17 §17.6.3)
- **Keypoints:** left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle

#### Regra 2.3 — Trabalho com braços elevados (`elevated_arm_work`)
- **Condição:** `shoulder_elevation_L < 90°` **OU** `shoulder_elevation_R < 90°`
- **Justificativa:** braço acima da linha dos ombros por tempo prolongado causa fadiga muscular e lesão por esforço repetitivo (NR-17 §17.6.1)
- **Keypoints:** left_hip, left_shoulder, left_elbow (idem para lado direito)

#### Regra 2.4 — REBA ≥ 8 (`reba_<score>`)
- **Condição:** `reba.final_score >= 8`
- **Justificativa:** Nível de Ação 4 (intervenção imediata) do instrumento REBA
- **Keypoints:** todos os keypoints usados pelo motor REBA

**Histerese:** entrada requer 5 frames consecutivos; saída requer 15 frames consecutivos.

---

### Categoria 3 — RISCO IMEDIATO
**Definição:** evento súbito com potencial de lesão grave. **Recall ≥ 0,90 é obrigatório** — falso negativo é inaceitável; ajustar thresholds para favorece recall sobre precisão.

#### Regra 3.1 — Pessoa caída (`person_fallen`)
- **Condição:** `bbox_width / bbox_height > 1.5`
- **Justificativa:** bounding box significativamente mais largo que alto indica pessoa deitada no chão
- **Keypoints:** bbox derivada do detector de pose

#### Regra 3.2 — Risco de instabilidade (`balance_loss_risk`)
- **Condição:** `trunk_rotation_proxy > 0.35`
- **Justificativa:** grande assimetria horizontal entre ombros e quadris, normalizada pela altura, indica torção repentina ou perda de equilíbrio
- **Keypoints:** mid_shoulder, mid_hip, bbox_height

**Histerese (recall-optimised):** entrada requer apenas **2 frames consecutivos**; saída requer 15 frames. Threshold baixo prioriza recall.

---

### Categoria 4 — APROXIMAÇÃO A ZONA CRÍTICA
**Definição:** tornozelos do trabalhador detectados dentro ou nas proximidades de uma zona demarcada.

#### Regra 4.1 — Invasão de zona (`critical_zone_entry`)
- **Condição:** `cv2.pointPolygonTest(zone_polygon, ankle_point) >= 0`
- **Keypoints:** left_ankle, right_ankle
- **Severidade:** determinada pelo atributo `severity` da zona (1–4)

#### Regra 4.2 — Aproximação de zona (`zone_approaching`)
- **Condição:** distância de qualquer tornozelo à borda da zona `< 80 px` (configurável)
- **Cruzamento com Fase 1:** se a zona exige EPI específico (`required_ppe`) e o operador está sem ele, severidade é elevada em +1

---

## 5. Máquina de Estados por Pessoa

```
          ┌─────────────┐
          │    NORMAL   │◄──── EXIT_FRAMES (15) sem detecção
          └──────┬──────┘
                 │ ENTER_FRAMES (5) ou ENTER_RISK (2)
    ┌────────────┼────────────────────┐
    ▼            ▼                   ▼
ERGONOMIC    IMMEDIATE_RISK     CRITICAL_ZONE
INADEQUATE   (enter=2 frames)
```

Cada pessoa rastreada mantém estado independente via `person_id` do ByteTrack.  
IDs são efêmeros (não persistem entre sessões) — conformidade LGPD.

---

## 6. Limitações Declaradas e Plano de Evolução

| Limitação | Impacto | Mitigação Atual | Evolução Fase 4 |
|-----------|---------|-----------------|-----------------|
| Visão 2D monocular | Ângulos fora do plano são invisíveis | Proxy de rotação de tronco | Câmera estéreo / multi-câmera |
| Carga e acoplamento não detectáveis | Score REBA subestimado | Valores conservadores documentados | Sensor de força / IMU |
| Desvio de punho não detectável em 2D | Grupo B subestimado | Punho fixo em score 1 | Hand pose estimation |
| Queda em câmera frontal pode não inverter bbox | Recall de queda comprometido | Regra de velocidade vertical na Etapa 6 | Fusão multi-sensor |
| Oclusão por equipamentos industriais | Keypoints perdidos | Descarte de ângulo (não interpolação) | Re-ID multi-câmera |
