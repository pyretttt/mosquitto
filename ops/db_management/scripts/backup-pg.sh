#!/usr/bin/env bash
# Host-side helper: run pg_dump via kubectl exec (good for first backup before
# you finish the Job/CronJob track). Prefer the in-cluster Job for TASKS.md §3.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NS_DB="${NS_DB:-records}"
PG_HELM_RELEASE="${PG_HELM_RELEASE:-records-pg}"
POSTGRES_DB="${POSTGRES_DB:-records}"
POSTGRES_USER="${POSTGRES_USER:-records}"
OUT_DIR="${ROOT}/backups"
STAMP="$(date +%Y%m%d%H%M%S)"
OUT_FILE="${OUT_DIR}/records-${STAMP}.dump"

mkdir -p "$OUT_DIR"

SECRET_NAME="${PG_HELM_RELEASE}-postgresql"

POD="$(kubectl -n "$NS_DB" get pods -l app.kubernetes.io/name=postgresql \
  -o jsonpath='{.items[0].metadata.name}')"

# Prefer custom-user key; fall back to postgres-password (Bitnami chart versions differ).
PASS="$(kubectl -n "$NS_DB" get secret "$SECRET_NAME" -o jsonpath='{.data.password}' 2>/dev/null | base64 -d || true)"
if [[ -z "$PASS" ]]; then
  PASS="$(kubectl -n "$NS_DB" get secret "$SECRET_NAME" -o jsonpath='{.data.postgres-password}' | base64 -d)"
fi

# TODO(you): replace this exec approach with the Job + ServiceAccount path — TASKS.md §3
echo "Dumping from pod/$POD → $OUT_FILE"
kubectl -n "$NS_DB" exec "$POD" -- \
  bash -ec "PGPASSWORD=$(printf %q "$PASS") pg_dump -U \"$POSTGRES_USER\" -d \"$POSTGRES_DB\" -Fc" \
  > "$OUT_FILE"

ls -la "$OUT_FILE"
echo "Restore later with: mise run restore-pg FILE=$OUT_FILE"
