#!/usr/bin/env bash
# Create a Secret from env / literals and show base64 encoding.
# Usage: DEMO_API_TOKEN=... DEMO_DB_PASSWORD=... bash scripts/create-plain-secret.sh
set -euo pipefail

NS="${NS_APP:-demo}"
NAME="${SECRET_NAME:-demo-app-secret}"
TOKEN="${DEMO_API_TOKEN:?set DEMO_API_TOKEN}"
PASS="${DEMO_DB_PASSWORD:-lab-db-pass}"

kubectl -n "$NS" create secret generic "$NAME" \
  --from-literal=API_TOKEN="$TOKEN" \
  --from-literal=DB_PASSWORD="$PASS" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "--- Secret .data (base64, not encryption) ---"
kubectl -n "$NS" get secret "$NAME" -o jsonpath='{.data.API_TOKEN}'
echo
echo "Decode with: kubectl -n $NS get secret $NAME -o jsonpath='{.data.API_TOKEN}' | base64 -d; echo"
