#!/usr/bin/env bash
# Sobe os 3 projetos do Cypher localmente com a empresa `demo`, seguindo
# docs/e1-runbook-local.md e docs/empresa-demo-runbook.md do cypher-front.
#
# Este mesmo arquivo é mantido em cópia idêntica nos 3 repositórios
# (cypher-front, cypher, cypher-capture) — pode ser rodado a partir de
# qualquer um deles. Ele só orquestra processos (mvnw, npm, venv/python);
# não precisa dos outros dois repositórios estarem no PATH, só nos
# caminhos configurados abaixo (padrão: layout usado neste projeto).
#
# Uso:
#   ./scripts/subir-empresa-demo.sh start   [--no-seed] [--orchestrator] [--device cuda|cpu] [--source <video|índice>]
#   ./scripts/subir-empresa-demo.sh status
#   ./scripts/subir-empresa-demo.sh stop
#   ./scripts/subir-empresa-demo.sh logs [backend|front|capture]
#
# "start" sobe backend → (semeia empresa demo, se ainda não tiver rodado) →
# front → um pipeline de captura (CAM-WEBCAM, ver empresa-demo-runbook.md §4)
# lendo --source (ex.: 0 para a webcam) ou, por padrão, o vídeo de demonstração
# versionado em cypher-capture/data/demo/. Com --orchestrator,
# sobe o orquestrador de
# processos (reconcilia todo o catálogo, inclusive as câmeras RTSP reais —
# exige CYPHER_RTSP_USER/CYPHER_RTSP_PASS num .env do cypher-capture).
#
# LEMBRETE — .env do cypher-capture não é carregado automaticamente por
# run_pipeline.py nem pelo orquestrador (nenhum dos dois chama load_dotenv();
# só src/utils/upload.py faz isso, e tarde demais pro uploader do Cloudinary
# ser escolhido). Este script já resolve isso sozinho desde 2026-10-21: o
# "start" sempre faz `source "$CAPTURE_DIR/.env"` antes de decidir device ou
# subir qualquer processo — então rodando via este script, Cloudinary e
# credenciais RTSP (CYPHER_RTSP_USER/PASS, CYPHER_<CAMERA>_USER/PASS) sempre
# chegam ao pipeline. O lembrete só importa se você subir run_pipeline.py OU
# src.orchestrator NA MÃO, fora deste script — aí ainda precisa de:
#   cd cypher-capture && set -a && source .env && set +a
# (achado 2026-10-20, ao cadastrar a câmera CAM-TEATRO1/VIPC-1230-B-G2 do
# ambiente simulado em teatro: 29 incidentes saíram com snapshotUrl null até
# perceber que faltava isso.)
#
# CUDA: "start" já detecta e usa GPU automaticamente quando o PyTorch do
# venv enxerga uma (sem precisar passar --device) — cai pra CPU sozinho em
# máquina sem GPU. Não hardcodamos "cuda" aqui de propósito, pra o script
# continuar funcionando em qualquer notebook da equipe.
#
# Onde estão os repositórios: por padrão, as pastas vizinhas da que contém este
# script (clone os 3 lado a lado: cypher-front, cypher-capture e o backend, que
# pode se chamar cypher, Cypher-Api ou cypher-api). Só é preciso configurar as
# variáveis abaixo se você clonou em outro layout.
#
# Variáveis de ambiente (todas opcionais, com default):
#   CYPHER_BACKEND_DIR   default: pasta vizinha cypher | Cypher-Api | cypher-api
#   CYPHER_FRONT_DIR     default: pasta vizinha cypher-front
#   CYPHER_CAPTURE_DIR   default: pasta vizinha cypher-capture
#   CYPHER_RUN_DIR       default: $HOME/.cypher-demo (pids, logs e marcador do seed)
#   CYPHER_DEMO_GESTOR_PASSWORD    default: demo12345
#   CYPHER_DEMO_OPERADOR_PASSWORD  default: operador123
#   CYPHER_API_URL        default: http://localhost:8080/api/v1
#   CYPHER_GESTOR_USER     default: gestor@demo   (conta de serviço usada pelo pipeline/orquestrador)
#   CYPHER_GESTOR_PASSWORD default: $CYPHER_DEMO_GESTOR_PASSWORD

set -uo pipefail

# Os 3 repositórios ficam lado a lado; este script está em <repo>/scripts/.
PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

first_dir() {
  # imprime o primeiro diretório existente entre os candidatos
  local d
  for d in "$@"; do
    if [ -d "$d" ]; then echo "$d"; return 0; fi
  done
  echo "$1"
}

BACKEND_DIR="${CYPHER_BACKEND_DIR:-$(first_dir "$PARENT_DIR/cypher" "$PARENT_DIR/Cypher-Api" "$PARENT_DIR/cypher-api")}"
FRONT_DIR="${CYPHER_FRONT_DIR:-$PARENT_DIR/cypher-front}"
CAPTURE_DIR="${CYPHER_CAPTURE_DIR:-$PARENT_DIR/cypher-capture}"

API_URL="${CYPHER_API_URL:-http://localhost:8080/api/v1}"
GESTOR_PASSWORD="${CYPHER_DEMO_GESTOR_PASSWORD:-demo12345}"
OPERADOR_PASSWORD="${CYPHER_DEMO_OPERADOR_PASSWORD:-operador123}"
export CYPHER_GESTOR_USER="${CYPHER_GESTOR_USER:-gestor@demo}"
export CYPHER_GESTOR_PASSWORD="${CYPHER_GESTOR_PASSWORD:-$GESTOR_PASSWORD}"

RUN_DIR="${CYPHER_RUN_DIR:-$HOME/.cypher-demo}"
mkdir -p "$RUN_DIR"
BACKEND_PID_FILE="$RUN_DIR/backend.pid"
FRONT_PID_FILE="$RUN_DIR/front.pid"
CAPTURE_PID_FILE="$RUN_DIR/capture.pid"
SEED_MARKER="$RUN_DIR/seed-demo.done"
BACKEND_LOG="$RUN_DIR/backend.log"
FRONT_LOG="$RUN_DIR/front.log"
CAPTURE_LOG="$RUN_DIR/capture.log"

log()  { echo "[subir-empresa-demo] $*"; }
warn() { echo "[subir-empresa-demo] AVISO: $*" >&2; }
die()  { echo "[subir-empresa-demo] ERRO: $*" >&2; exit 1; }

is_running() {
  # $1 = pid file
  [ -f "$1" ] && kill -0 "-$(cat "$1")" 2>/dev/null
}

start_bg() {
  # $1 = pid file, $2 = log file, $3 = diretório, resto = comando
  local pidfile="$1" logfile="$2" dir="$3"; shift 3
  if is_running "$pidfile"; then
    log "já rodando (pid $(cat "$pidfile")), pulando: $*"
    return 0
  fi
  ( cd "$dir" && exec setsid "$@" >"$logfile" 2>&1 ) &
  echo $! >"$pidfile"
  sleep 0.3
}

stop_one() {
  # $1 = nome, $2 = pid file
  local nome="$1" pidfile="$2"
  if is_running "$pidfile"; then
    local pid; pid="$(cat "$pidfile")"
    log "derrubando $nome (grupo de processo $pid)"
    kill -TERM "-$pid" 2>/dev/null
    sleep 1
    kill -KILL "-$pid" 2>/dev/null
  else
    log "$nome não estava rodando"
  fi
  rm -f "$pidfile"
}

wait_for_health() {
  local tries=0
  log "aguardando backend responder em $API_URL/health ..."
  until curl -sf "$API_URL/health" >/dev/null 2>&1; do
    tries=$((tries + 1))
    if [ "$tries" -gt 60 ]; then
      die "backend não respondeu em 60s — veja $BACKEND_LOG"
    fi
    sleep 1
  done
  log "backend no ar."
}

cmd_start() {
  local no_seed=0 use_orchestrator=0 no_capture=0 device="" source_arg=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --no-seed) no_seed=1 ;;
      --orchestrator) use_orchestrator=1 ;;
      --no-capture) no_capture=1 ;;
      --device) device="$2"; shift ;;
      --source) source_arg="$2"; shift ;;
      *) die "flag desconhecida: $1" ;;
    esac
    shift
  done

  [ -d "$BACKEND_DIR" ] || die "backend não encontrado em $BACKEND_DIR — clone ao lado deste repositório ou defina CYPHER_BACKEND_DIR"
  [ -d "$FRONT_DIR" ] || die "front não encontrado em $FRONT_DIR — defina CYPHER_FRONT_DIR"
  [ -d "$CAPTURE_DIR" ] || die "capture não encontrado em $CAPTURE_DIR — defina CYPHER_CAPTURE_DIR"

  # Carrega o .env do capture ANTES de decidir device ou subir qualquer
  # processo: resolve de vez o lembrete do cabeçalho (Cloudinary e
  # credenciais RTSP deixavam de chegar ao pipeline quando o shell que
  # rodava este script não tinha o .env já exportado). Com isso o "start"
  # deste script sempre sobe com Cloudinary configurado, se o .env existir.
  if [ -f "$CAPTURE_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$CAPTURE_DIR/.env"
    set +a
  else
    warn "$CAPTURE_DIR/.env não existe — subindo sem Cloudinary/credenciais RTSP"
  fi

  # Python do capture: venv do projeto se existir, senão o do sistema.
  local py="./venv/bin/python"
  [ -x "$CAPTURE_DIR/venv/bin/python" ] || py="python3"

  # Device: usa CUDA só se o PyTorch instalado no venv enxergar a GPU; senão CPU
  # (mais lento, mas funciona em qualquer máquina).
  if [ -z "$device" ]; then
    if (cd "$CAPTURE_DIR" && "$py" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)") >/dev/null 2>&1; then
      device="cuda"
    else
      device="cpu"
      log "GPU não disponível para o PyTorch — usando --device cpu (a detecção roda em poucos FPS, é esperado)"
    fi
  fi

  # Fonte de vídeo: --source (ex.: 0 para a webcam) ou, por padrão, o vídeo de
  # demonstração versionado — assim o resultado não depende de ter webcam.
  if [ -z "$source_arg" ]; then
    source_arg="$CAPTURE_DIR/data/demo/validacao-camera-2026-10-14.mp4"
    log "fonte padrão: vídeo de demonstração (use --source 0 para a webcam)"
  fi

  log "1/4 — backend ($BACKEND_DIR)"
  start_bg "$BACKEND_PID_FILE" "$BACKEND_LOG" "$BACKEND_DIR" ./mvnw spring-boot:run
  wait_for_health

  if [ "$no_seed" -eq 0 ] && [ ! -f "$SEED_MARKER" ]; then
    log "2/4 — semeando empresa demo (câmeras reais + equipes de teste)"
    if (cd "$BACKEND_DIR" && CYPHER_DEMO_GESTOR_PASSWORD="$GESTOR_PASSWORD" CYPHER_DEMO_OPERADOR_PASSWORD="$OPERADOR_PASSWORD" ./scripts/seed-empresa-demo.sh); then
      touch "$SEED_MARKER"
    else
      warn "seed falhou ou já tinha sido rodado antes — veja docs/empresa-demo-runbook.md antes de rodar de novo (histórico de gamificação não é idempotente)"
    fi
  else
    log "2/4 — seed pulado (--no-seed ou já rodado antes: $SEED_MARKER)"
  fi

  log "3/4 — front ($FRONT_DIR)"
  start_bg "$FRONT_PID_FILE" "$FRONT_LOG" "$FRONT_DIR" npm start

  log "4/4 — capture ($CAPTURE_DIR)"
  if [ "$no_capture" -eq 1 ]; then
    log "capture pulado (--no-capture) — suba manualmente com run_pipeline.py se precisar de múltiplas câmeras/vídeos de validação"
  elif [ "$use_orchestrator" -eq 1 ]; then
    log "modo orquestrador — reconcilia todo o catálogo (precisa de .env com credenciais RTSP reais para as câmeras da meta indústria)"
    start_bg "$CAPTURE_PID_FILE" "$CAPTURE_LOG" "$CAPTURE_DIR" \
      "$py" -m src.orchestrator --api-url "$API_URL"
  else
    log "modo câmera única CAM-WEBCAM, fonte: $source_arg (ver empresa-demo-runbook.md §4)"
    start_bg "$CAPTURE_PID_FILE" "$CAPTURE_LOG" "$CAPTURE_DIR" \
      "$py" src/run_pipeline.py \
        --source "$source_arg" --device "$device" --no-display \
        --camera-id CAM-WEBCAM --api-url "$API_URL"
  fi

  log "pronto. Front em http://localhost:4200 — login gestor@demo / \$CYPHER_DEMO_GESTOR_PASSWORD"
  cmd_status
}

cmd_status() {
  for par in "backend:$BACKEND_PID_FILE" "front:$FRONT_PID_FILE" "capture:$CAPTURE_PID_FILE"; do
    local nome="${par%%:*}" pidfile="${par#*:}"
    if is_running "$pidfile"; then
      echo "$nome: rodando (pid $(cat "$pidfile"))"
    else
      echo "$nome: parado"
    fi
  done
}

cmd_stop() {
  stop_one "capture" "$CAPTURE_PID_FILE"
  stop_one "front" "$FRONT_PID_FILE"
  stop_one "backend" "$BACKEND_PID_FILE"
}

cmd_logs() {
  case "${1:-}" in
    backend) tail -f "$BACKEND_LOG" ;;
    front) tail -f "$FRONT_LOG" ;;
    capture) tail -f "$CAPTURE_LOG" ;;
    *) tail -f "$BACKEND_LOG" "$FRONT_LOG" "$CAPTURE_LOG" ;;
  esac
}

case "${1:-}" in
  start) shift; cmd_start "$@" ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  logs) shift; cmd_logs "$@" ;;
  *) die "uso: $0 {start|stop|status|logs} [flags] — ver cabeçalho do script" ;;
esac
