#!/usr/bin/env bash
# Compare ConfigMap vs Secret: both are plain YAML in etcd unless encryption-at-rest is on.
set -euo pipefail

NS="${NS_APP:-demo}"

echo "=== ConfigMap (plaintext in .data) ==="
kubectl -n "$NS" get configmap demo-app-config -o yaml | sed -n '/^data:/,/^[^ ]/p' || true

echo
echo "=== Secret (base64 in .data — still reversible) ==="
if kubectl -n "$NS" get secret demo-app-secret >/dev/null 2>&1; then
  kubectl -n "$NS" get secret demo-app-secret -o yaml | sed -n '/^data:/,/^[^ ]/p'
  echo
  echo "Decoded API_TOKEN:"
  kubectl -n "$NS" get secret demo-app-secret -o jsonpath='{.data.API_TOKEN}' | base64 -d
  echo
else
  echo "demo-app-secret not found — run: mise run secret-create"
fi
