#!/usr/bin/env bash
# Download a ResNet50 ONNX model into the Triton model repository.
#
# Source: ONNX model zoo (ResNet50 v1.7, opset 12, NCHW).
# This is idempotent — running it twice is a no-op.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/model_repository/resnet50/1"
TARGET_FILE="${TARGET_DIR}/model.onnx"
URL="${RESNET50_ONNX_URL:-https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx}"

mkdir -p "${TARGET_DIR}"

if [[ -s "${TARGET_FILE}" ]]; then
  echo "[fetch_model] resnet50 ONNX already present at ${TARGET_FILE}"
  exit 0
fi

echo "[fetch_model] Downloading ResNet50 ONNX from ${URL}"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 3 -o "${TARGET_FILE}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${TARGET_FILE}" "${URL}"
else
  echo "[fetch_model] need curl or wget" >&2
  exit 1
fi

echo "[fetch_model] Wrote $(du -h "${TARGET_FILE}" | cut -f1) to ${TARGET_FILE}"
