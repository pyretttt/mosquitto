#!/usr/bin/env bash
# Generate a self-signed TLS cert for INGRESS_HOST and create Secret records-tls.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NS_APP="${NS_APP:-records}"
INGRESS_HOST="${INGRESS_HOST:-records.localtest.me}"
CERT_DIR="$ROOT/certs"

mkdir -p "$CERT_DIR"

openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout "$CERT_DIR/tls.key" \
  -out "$CERT_DIR/tls.crt" \
  -subj "/CN=${INGRESS_HOST}" \
  -addext "subjectAltName=DNS:${INGRESS_HOST}"

cp "$CERT_DIR/tls.crt" "$CERT_DIR/ca.crt"

kubectl get ns "$NS_APP" >/dev/null 2>&1 || kubectl apply -f "$ROOT/k8s/namespace.yaml"

kubectl -n "$NS_APP" create secret tls records-tls \
  --cert="$CERT_DIR/tls.crt" \
  --key="$CERT_DIR/tls.key" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Wrote $CERT_DIR/tls.{crt,key} and applied Secret records-tls in $NS_APP"
echo "Trust for curl: --cacert $CERT_DIR/ca.crt"
