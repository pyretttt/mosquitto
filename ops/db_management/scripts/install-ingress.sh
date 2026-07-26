#!/usr/bin/env bash
# Install ingress-nginx for kind (controller watches ingress-ready nodes).
set -euo pipefail

kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.12.0/deploy/static/provider/kind/deploy.yaml

echo "Waiting for ingress-nginx controller..."
kubectl -n ingress-nginx wait --for=condition=Available deploy/ingress-nginx-controller --timeout=180s
echo "ingress-nginx ready"
