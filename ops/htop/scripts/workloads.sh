#!/usr/bin/env bash
# Start / stop / status for synthetic workloads.
# Processes are tagged via LAB_TAG env and logged under .run/
#
# Usage:
#   scripts/workloads.sh start [cpu|mem|io|fork|hung|all]
#   scripts/workloads.sh stop
#   scripts/workloads.sh status
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT/.run}"
DATA_DIR="${DATA_DIR:-$ROOT/.data}"
LAB_TAG="${LAB_TAG:-htop-lab}"
export LAB_TAG DATA_DIR RUN_DIR

mkdir -p "$RUN_DIR" "$DATA_DIR"

resolve_python() {
  if command -v python >/dev/null 2>&1; then
    command -v python
  else
    command -v python3
  fi
}

start_one() {
  local name="$1"
  local script="$2"
  local pidfile="$RUN_DIR/${name}.pid"
  local logfile="$RUN_DIR/${name}.log"
  local python
  python="$(resolve_python)"

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "$name already running pid=$(cat "$pidfile")"
    return 0
  fi

  # TODO(you): start with lower priority via nice — see TASKS.md §5
  nohup env LAB_TAG="$LAB_TAG" DATA_DIR="$DATA_DIR" HTOP_LAB_ROLE="$name" \
    "$python" "$ROOT/workloads/$script" >"$logfile" 2>&1 &
  echo $! >"$pidfile"
  echo "started $name pid=$(cat "$pidfile") (log: $logfile)"
}

start_cpu()  { start_one cpu  cpu_burn.py; }
start_mem()  { start_one mem  mem_hog.py; }
start_io()   { start_one io   io_churn.py; }
start_fork() { start_one fork fork_tree.py; }
start_hung() { start_one hung hung_worker.py; }

start_all() {
  start_cpu
  start_mem
  start_io
  start_fork
  start_hung
}

stop_all() {
  local stopped=0
  if [[ -d "$RUN_DIR" ]]; then
    for pidfile in "$RUN_DIR"/*.pid; do
      [[ -e "$pidfile" ]] || continue
      local pid
      pid="$(cat "$pidfile")"
      if kill -0 "$pid" 2>/dev/null; then
        # kill process group-ish: parent first; children of fork may linger
        kill "$pid" 2>/dev/null || true
        # fork tree children
        pkill -P "$pid" 2>/dev/null || true
        echo "sent SIGTERM to $pid ($(basename "$pidfile" .pid))"
        stopped=1
      fi
      rm -f "$pidfile"
    done
  fi
  # belt-and-suspenders: anything still matching lab env in cmdline/logs
  pkill -f "workloads/(cpu_burn|mem_hog|io_churn|fork_tree|hung_worker)\\.py" 2>/dev/null || true
  if [[ "$stopped" -eq 0 ]]; then
    echo "no tracked workloads (or already stopped)"
  fi
}

status() {
  echo "=== pidfiles in $RUN_DIR ==="
  local any=0
  for pidfile in "$RUN_DIR"/*.pid; do
    [[ -e "$pidfile" ]] || continue
    any=1
    local name pid
    name="$(basename "$pidfile" .pid)"
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "RUNNING  $name  pid=$pid"
    else
      echo "STALE    $name  pid=$pid"
    fi
  done
  [[ "$any" -eq 1 ]] || echo "(none)"
  echo
  echo "=== pgrep hint ==="
  # TODO(you): replace with your preferred pgrep/htop filter — see TASKS.md §1
  pgrep -fl "cpu_burn|mem_hog|io_churn|fork_tree|hung_worker" || echo "(no matches)"
}

cmd="${1:-status}"
target="${2:-all}"

case "$cmd" in
  start)
    case "$target" in
      all)  start_all ;;
      cpu)  start_cpu ;;
      mem)  start_mem ;;
      io)   start_io ;;
      fork) start_fork ;;
      hung) start_hung ;;
      *)
        echo "unknown workload: $target (cpu|mem|io|fork|hung|all)" >&2
        exit 1
        ;;
    esac
    ;;
  stop)   stop_all ;;
  status) status ;;
  *)
    echo "usage: $0 {start|stop|status} [cpu|mem|io|fork|hung|all]" >&2
    exit 1
    ;;
esac
