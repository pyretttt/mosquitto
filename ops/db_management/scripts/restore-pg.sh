#!/usr/bin/env bash
# Restore a custom-format dump produced by backup-pg.sh into the lab Postgres.
set -euo pipefail

NS_DB="${NS_DB:-records}"
POSTGRES_DB="${POSTGRES_DB:-records}"
POSTGRES_USER="${POSTGRES_USER:-records}"
FILE="${FILE:?set FILE=/path/to/records-....dump}"

if [[ ! -f "$FILE" ]]; then
  echo "FILE not found: $FILE" >&2
  exit 1
fi

PG_HELM_RELEASE="${PG_HELM_RELEASE:-records-pg}"
SECRET_NAME="${PG_HELM_RELEASE}-postgresql"

POD="$(kubectl -n "$NS_DB" get pods -l app.kubernetes.io/name=postgresql \
  -o jsonpath='{.items[0].metadata.name}')"

PASS="$(kubectl -n "$NS_DB" get secret "$SECRET_NAME" -o jsonpath='{.data.password}' 2>/dev/null | base64 -d || true)"
if [[ -z "$PASS" ]]; then
  PASS="$(kubectl -n "$NS_DB" get secret "$SECRET_NAME" -o jsonpath='{.data.postgres-password}' | base64 -d)"
fi

# TODO(you): after NetworkPolicies, restore may need to run from a labeled Job — TASKS.md §3–4
echo "Restoring $FILE into pod/$POD db=$POSTGRES_DB"
kubectl -n "$NS_DB" exec -i "$POD" -- \
  bash -ec "PGPASSWORD=$(printf %q "$PASS") pg_restore -U \"$POSTGRES_USER\" -d \"$POSTGRES_DB\" --clean --if-exists" \
  < "$FILE"

echo "Restore finished. Verify: curl records via port-forward or Ingress"
