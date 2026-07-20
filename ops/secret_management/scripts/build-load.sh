#!/usr/bin/env bash
# Build demo image and load into kind.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER_NAME="${CLUSTER_NAME:-secrets-lab}"
IMAGE_NAME="${IMAGE_NAME:-demo-app}"
IMAGE_TAG="${IMAGE_TAG:-local}"

docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" "$ROOT/app"
kind load docker-image "${IMAGE_NAME}:${IMAGE_TAG}" --name "$CLUSTER_NAME"
echo "Loaded ${IMAGE_NAME}:${IMAGE_TAG} into kind/$CLUSTER_NAME"
