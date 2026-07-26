#!/usr/bin/env bash
# Install Bitnami PostgreSQL into the lab namespace.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NS_DB="${NS_DB:-records}"
PG_HELM_RELEASE="${PG_HELM_RELEASE:-records-pg}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:?set POSTGRES_PASSWORD in .env}"
POSTGRES_USER="${POSTGRES_USER:-records}"
POSTGRES_DB="${POSTGRES_DB:-records}"

kubectl get ns "$NS_DB" >/dev/null 2>&1 || kubectl apply -f "$ROOT/k8s/namespace.yaml"

helm upgrade --install "$PG_HELM_RELEASE" oci://registry-1.docker.io/bitnamicharts/postgresql \
  -n "$NS_DB" \
  -f "$ROOT/charts/postgresql/values-local.yaml" \
  --set auth.username="$POSTGRES_USER" \
  --set auth.database="$POSTGRES_DB" \
  --set auth.password="$POSTGRES_PASSWORD" \
  --set auth.postgresPassword="$POSTGRES_PASSWORD" \
  --wait --timeout 5m

echo "Postgres release '$PG_HELM_RELEASE' ready in ns/$NS_DB"
echo "Secret: kubectl -n $NS_DB get secret ${PG_HELM_RELEASE}-postgresql -o yaml"
