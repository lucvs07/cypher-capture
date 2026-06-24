# Prontidão para a Segunda Mentoria — Dia 3 (24/06/2026)

**Mentores:** Fernando V. Marcolina e Wendel de Almeida Passos · **Data:** quinta 25/06/2026.
**Verificação real:** `venv` Python 3.10, CUDA, commit `fe4dc54`, working tree limpo.

## Esta equipe está pronta para a mentoria?

| # | Critério | Status | Evidência / pendência |
|---|---|---|---|
| 1 | Código funciona (testes, CLI, vídeos) | ✅ | `pytest tests/` → **89 passed in 2,71s**; `run_pipeline.py --help` ok; clipes rodam e geram `*_annot.mp4` |
| 2 | Métricas medidas e documentadas | ✅ | Recall IR 0,986 · Latência GPU p95 44,9 ms · PCK/acurácia/falso-alarme em [metricas-dia2.md](metricas-dia2.md) |
| 3 | Documentação técnica clara (README) | ✅ | README PT com seção Fase 2, CLI, zonas, testes, avaliação, latência, limitações 2D |
| 4 | Esqueleto de apresentação pronto | ✅ | [apresentacao-esqueleto.md](apresentacao-esqueleto.md) — input pronto para Claude Design |
| 5 | Perguntas técnicas para os mentores | ⚠️ | Esboço abaixo — **revisar com a equipe** |
| 6 | Cada membro sabe falar sobre sua parte | ⚠️ | Papéis ainda em `[PREENCHER]` no esqueleto §1 — **definir antes de quinta** |

## Veredito

> **PRONTO, com 2 ações de equipe (não-técnicas) antes de quinta.** O core da Fase 2 está
> **validado, documentado e testado**: recall de risco imediato 0,986 (meta 0,90), latência
> 45 ms em GPU, 89 testes verdes. A apresentação para o Claude Design **pode começar
> imediatamente**. Faltam apenas dois itens de organização da equipe:
>
> 1. **Definir e ensaiar papéis** — preencher `[PREENCHER]` em [apresentacao-esqueleto.md](apresentacao-esqueleto.md) §1
>    e garantir que todos os 6 membros falem. *(Equipe, ~30 min)*
> 2. **Fechar as perguntas para os mentores** — validar a lista abaixo. *(Equipe, ~15 min)*
>
> Opcional, eleva a demo (não bloqueia): refilmar o cenário de ergonomia **de perfil** (subir
> acurácia acima de 0,85). **O cenário de zona crítica não será demonstrado nesta mentoria**
> (funcionalidade pronta no código; validação fica para pós-mentoria).

## Perguntas técnicas sugeridas para Fernando e Wendel

1. **Hardware da Metaindústria:** há GPU disponível por câmera, ou o deploy será multiplexado
   numa GPU central? (a latência só passa em GPU)
2. **Precisão vs recall:** em segurança, mantemos recall 0,99 com precisão 0,42–0,58 (falsos
   alarmes toleráveis), ou há um teto de falso-alarme operacional que devemos respeitar?
3. **Violação de EPI:** vale re-treinar `exp.pt` com classes `no_*`, ou inferir violação por
   ausência ("Person sem HELMET associado")?
4. **Ângulo de câmera:** a limitação 2D (flexão de tronco só de perfil) sugere câmeras laterais.
   Isso é viável no layout da Metaindústria?
5. **PCK independente:** vale o esforço de rotular keypoints à mão (CVAT) para um PCK válido, ou
   a acurácia de classificação já é a métrica que importa para a banca?

## Como reproduzir a verificação

```bash
source venv/bin/activate
python -m pytest tests/ -q                       # 89 passed
python src/run_pipeline.py --help                # CLI + flags
python src/run_pipeline.py --source data/validation/cenario2.mp4 --device cuda \
  --output outputs/captures/cenario2_annot.mp4 --no-display
python scripts/evaluate_pose.py \
  --annotations data/annotations/cenario2_gt.json \
  --video data/validation/cenario2.mp4 --device cuda   # immediate_risk recall ~0,986
```
