#!/usr/bin/env bash
# Prod lifecycle (docs/40-docker-prod.md, docs/deploy.md):
#   prod.sh build | up | down | logs | status | migrate | backup | restore <archive> <bundle> <dir> | start | stop
# Env: DATA (default ~/docker-data/voicebot), TAG (image tag; default latest), PROD_PORT (default 8010),
#      PROJECT (compose project; default voicebot), BACKUP_DIR (default ~/docker-data/backups),
#      KEEP (backups kept, 7), MIN_FREE_MB (2048), MAX_TOTAL_MB (5120), FORCE=1 (back up even if the vault head is unchanged).
# start/stop delegate to scripts/voicebot (docs/41-service-health.md), which also brings up oMLX/ASR/colima.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)

DATA=${DATA:-$HOME/docker-data/voicebot}
PROD_PORT=${PROD_PORT:-8010}
PROJECT=${PROJECT:-voicebot}
BACKUP_DIR=${BACKUP_DIR:-$HOME/docker-data/backups}
KEEP=${KEEP:-7}
MIN_FREE_MB=${MIN_FREE_MB:-2048}
MAX_TOTAL_MB=${MAX_TOTAL_MB:-5120}
TAG_ARG=${TAG:-}                # an explicit TAG wins over `latest` (build tags only that)
export DATA PROD_PORT TAG="${TAG:-latest}"

compose() { docker compose -p "$PROJECT" -f "$ROOT/docker-compose.yml" "$@"; }

wait_for() {  # wait_for <url> <seconds>
  local url=$1 deadline=$((SECONDS + $2))
  until curl -fsS "$url" > /dev/null 2>&1; do
    [ $SECONDS -lt "$deadline" ] || { echo "no answer from $url" >&2; return 1; }
    sleep 2
  done
}

cmd_build() {
  local tag=${TAG_ARG:-$(git -C "$ROOT" describe --always --dirty)}
  docker build --build-arg VOICEBOT_VERSION="$tag" -t "voicebot:$tag" "$ROOT"
  [ -n "$TAG_ARG" ] || docker tag "voicebot:$tag" voicebot:latest
  echo "built voicebot:$tag"
}

cmd_up() {
  [ -f "$DATA/.env" ] || { echo "$DATA/.env missing — run prod.sh migrate first" >&2; exit 1; }
  mkdir -p "$DATA/memory" "$DATA/secrets" "$DATA/logs"
  compose up -d
  echo "http://localhost:$PROD_PORT"
}

cmd_status() {
  compose ps
  curl -fsS "http://127.0.0.1:$PROD_PORT/api/config" && echo
}

# One-time: gather the checkout's state and the vault into $DATA. Refuses to overwrite anything.
# SRC is where the checkout kept its files (default: the repo root), WIKI_SRC the vault (~/llm-wiki).
cmd_migrate() {
  local src=${SRC:-$ROOT} wiki_src=${WIKI_SRC:-$HOME/llm-wiki}
  [ ! -e "$DATA" ] || { echo "refusing: $DATA already exists" >&2; exit 1; }
  mkdir -p "$DATA/memory" "$DATA/secrets" "$DATA/logs"
  move() {  # move <from> <to>
    [ -e "$1" ] || return 0
    [ ! -e "$2" ] || { echo "refusing: $2 already exists" >&2; exit 1; }
    mv "$1" "$2" && echo "moved $1 -> $2"
  }
  move "$wiki_src" "$DATA/llm-wiki"
  for f in memory.json memory_ops.jsonl shortmem.txt; do move "$src/$f" "$DATA/memory/$f"; done
  move "$src/settings.json" "$DATA/settings.json"
  for f in credentials.json token.json; do move "$src/$f" "$DATA/secrets/$f"; done
  move "$src/session.log" "$DATA/logs/session.log"
  # .env is copied, not moved: the dev checkout keeps running from the same secrets/config.
  if [ -f "$src/.env" ]; then cp "$src/.env" "$DATA/.env" && echo "copied $src/.env -> $DATA/.env"; fi
  echo "done: $DATA"
}

# Guards live here so manual and scheduled runs share them: lock, skip when the vault head is unchanged,
# rotate (leaving room for the new pair so KEEP is the total afterwards), free-space and total-size caps.
backup_log() {  # backup_log <ok|skip|fail> <text>
  mkdir -p "$DATA/logs"
  echo "$(date '+%F %T') | $1 | $2" >> "$DATA/logs/backup.log"
  tail -n 200 "$DATA/logs/backup.log" > "$DATA/logs/backup.log.tmp" && mv "$DATA/logs/backup.log.tmp" "$DATA/logs/backup.log"
}
backup_fail() { echo "backup: $1" >&2; backup_log fail "$1"; exit 1; }

cmd_backup() {
  local stamp head free total est pat
  stamp=$(date +%Y%m%d-%H%M%S)
  mkdir -p "$BACKUP_DIR"
  mkdir "$BACKUP_DIR/.lock" 2> /dev/null || backup_fail "another run holds $BACKUP_DIR/.lock"
  trap 'rmdir "$BACKUP_DIR/.lock"' EXIT
  head=$(git -C "$DATA/llm-wiki" rev-parse HEAD)
  if [ "${FORCE:-}" != 1 ] && [ "$(cat "$BACKUP_DIR/.last-head" 2> /dev/null)" = "$head" ]; then
    echo "backup: no new vault commit since last backup, skipping"; backup_log skip "head ${head:0:8} unchanged"; return 0
  fi
  rm -f "$BACKUP_DIR"/*.tmp   # leftovers of a killed run
  for pat in 'voicebot-data-*.tgz' 'llm-wiki-*.bundle'; do
    { ls -1 "$BACKUP_DIR"/$pat 2> /dev/null || true; } | sort -r | tail -n +"$KEEP" | while read -r f; do rm -f "$f"; done
  done
  free=$(df -Pm "$BACKUP_DIR" | awk 'NR==2 {print $4}')
  [ "$free" -ge "$MIN_FREE_MB" ] || backup_fail "only ${free} MB free in $BACKUP_DIR (need $MIN_FREE_MB)"
  total=$(du -sm "$BACKUP_DIR" | cut -f1); est=$(du -sm "$DATA" | cut -f1)
  [ $((total + est)) -le "$MAX_TOTAL_MB" ] || backup_fail "total ${total}+${est} MB would exceed MAX_TOTAL_MB=$MAX_TOTAL_MB"
  local tgz="$BACKUP_DIR/voicebot-data-$stamp.tgz" bundle="$BACKUP_DIR/llm-wiki-$stamp.bundle"
  git -C "$DATA/llm-wiki" bundle create "$bundle.tmp" --all
  tar czf "$tgz.tmp" -C "$DATA" --exclude='./llm-wiki/raw/.index' --exclude='./llm-wiki/.index' .
  mv "$bundle.tmp" "$bundle"; mv "$tgz.tmp" "$tgz"
  echo "$head" > "$BACKUP_DIR/.last-head"
  echo "$tgz"; echo "$bundle"
  backup_log ok "$tgz $bundle"
}

# restore <archive> <bundle> <dir>: untar into <dir>, verify the bundle, prove it serves on :8012.
cmd_restore() {
  local archive=$1 bundle=$2
  dir=$3   # global on purpose: the EXIT trap below outlives this function
  [ ! -e "$dir" ] || [ -z "$(ls -A "$dir")" ] || { echo "refusing: $dir is not empty" >&2; exit 1; }
  mkdir -p "$dir"
  tar xzf "$archive" -C "$dir"
  git bundle verify "$bundle"
  echo "restored vault head: $(git -C "$dir/llm-wiki" log -1 --oneline)"
  port=${RESTORE_PORT:-8012}
  DATA=$dir PROD_PORT=$port PROJECT="$PROJECT-restore" compose up -d
  trap 'DATA=$dir PROD_PORT=$port PROJECT="$PROJECT-restore" compose down > /dev/null 2>&1' EXIT
  wait_for "http://127.0.0.1:$port/api/config" 120
  curl -fsS "http://127.0.0.1:$port/api/config" && echo
  echo "restore ok: $dir"
}

case ${1:-} in
  build)   cmd_build ;;
  up)      cmd_up ;;
  down)    compose down ;;
  logs)    compose logs -f --tail=100 ;;
  status)  cmd_status ;;
  migrate) cmd_migrate ;;
  backup)  cmd_backup ;;
  restore) [ $# -eq 4 ] || { echo "usage: prod.sh restore <archive.tgz> <vault.bundle> <dir>" >&2; exit 1; }
           cmd_restore "$2" "$3" "$4" ;;
  start)   exec "$ROOT/scripts/voicebot" start "${@:2}" ;;
  stop)    exec "$ROOT/scripts/voicebot" stop "${@:2}" ;;
  *) echo "usage: prod.sh build|up|down|logs|status|migrate|backup|restore <archive> <bundle> <dir>|start|stop" >&2; exit 1 ;;
esac
