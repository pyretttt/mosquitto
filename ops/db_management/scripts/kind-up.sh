#!/usr/bin/env bash
# Create the kind cluster with Ingress port mappings.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER_NAME="${CLUSTER_NAME:-db-lab}"

if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  echo "kind cluster '$CLUSTER_NAME' already exists"
  exit 0
fi

kind create cluster --name "$CLUSTER_NAME" --config "$ROOT/kind/config.yaml"
echo "Created kind/$CLUSTER_NAME"
