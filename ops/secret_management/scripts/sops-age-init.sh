#!/usr/bin/env bash
# Generate age key for SOPS (private key stays local).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/local/age"
mkdir -p "$OUT_DIR"

if [[ -f "$OUT_DIR/key.txt" ]]; then
  echo "Key already exists at $OUT_DIR/key.txt"
else
  age-keygen -o "$OUT_DIR/key.txt"
  echo "Created $OUT_DIR/key.txt (gitignored via local/age/ — verify .gitignore)"
fi

PUB="$(grep -E '^# public key:' "$OUT_DIR/key.txt" | awk '{print $4}')"
echo "Public key: $PUB"
echo "TODO(you): put this public key in .sops.yaml (age: ...) — TASKS.md §3"
echo "Export for decrypt: export SOPS_AGE_KEY_FILE=$OUT_DIR/key.txt"
