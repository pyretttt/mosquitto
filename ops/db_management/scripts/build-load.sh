#!/usr/bin/env bash
# Build records-api image and load into kind.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER_NAME="${CLUSTER_NAME:-db-lab}"
IMAGE_NAME="${IMAGE_NAME:-records-api}"
IMAGE_TAG="${IMAGE_TAG:-local}"

docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" "$ROOT/app"
kind load docker-image "${IMAGE_NAME}:${IMAGE_TAG}" --name "$CLUSTER_NAME"
echo "Loaded ${IMAGE_NAME}:${IMAGE_TAG} into kind/$CLUSTER_NAME"
