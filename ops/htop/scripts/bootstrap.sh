#!/usr/bin/env bash
# Bootstrap lab dirs and sanity-check htop.
# TODO(you): install htop if missing — see TASKS.md §0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/.run" "$ROOT/.data"

if ! command -v htop >/dev/null 2>&1; then
  echo "htop not found on PATH."
  echo "  macOS:  brew install htop"
  echo "  Debian: sudo apt install htop"
  echo "Then re-run: mise run bootstrap"
  exit 1
fi

echo "ok: dirs ready under $ROOT"
htop --version
