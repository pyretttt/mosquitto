#!/usr/bin/env bash
# Helm install/upgrade records-api.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NS_APP="${NS_APP:-records}"
APP_HELM_RELEASE="${APP_HELM_RELEASE:-records-api}"
PG_HELM_RELEASE="${PG_HELM_RELEASE:-records-pg}"
POSTGRES_USER="${POSTGRES_USER:-records}"
POSTGRES_DB="${POSTGRES_DB:-records}"

kubectl get ns "$NS_APP" >/dev/null 2>&1 || kubectl apply -f "$ROOT/k8s/namespace.yaml"

# Bitnami secret keys vary slightly by chart version — detect password key.
SECRET_NAME="${PG_HELM_RELEASE}-postgresql"
PASSWORD_KEY="postgres-password"
if ! kubectl -n "$NS_APP" get secret "$SECRET_NAME" -o jsonpath="{.data.${PASSWORD_KEY}}" >/dev/null 2>&1; then
  PASSWORD_KEY="password"
fi

helm upgrade --install "$APP_HELM_RELEASE" "$ROOT/charts/records-api" \
  -n "$NS_APP" \
  -f "$ROOT/charts/records-api/values.yaml" \
  -f "$ROOT/charts/records-api/values-local.yaml" \
  --set postgres.host="${PG_HELM_RELEASE}-postgresql" \
  --set postgres.secretName="$SECRET_NAME" \
  --set postgres.passwordKey="$PASSWORD_KEY" \
  --set postgres.user="$POSTGRES_USER" \
  --set postgres.database="$POSTGRES_DB" \
  --wait --timeout 3m

echo "records-api deployed. Try: mise run port-forward && curl -s localhost:8080/records"
