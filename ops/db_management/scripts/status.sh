#!/usr/bin/env bash
# Quick status for the db_management lab.
set -euo pipefail

NS_APP="${NS_APP:-records}"

echo "== namespaces =="
kubectl get ns records ingress-nginx 2>/dev/null || true

echo "== pods ($NS_APP) =="
kubectl -n "$NS_APP" get pods,svc,ingress,networkpolicy,sa 2>/dev/null || true

echo "== ingress-nginx =="
kubectl -n ingress-nginx get pods,svc 2>/dev/null || true
