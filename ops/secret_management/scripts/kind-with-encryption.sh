#!/usr/bin/env bash
# Prepare EncryptionConfiguration and create a kind cluster with encryption-at-rest.
# TODO(you): understand each step — TASKS.md §2
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER_NAME="${CLUSTER_NAME:-secrets-lab}"
ENC_EXAMPLE="$ROOT/k8s/encryption/EncryptionConfiguration.yaml.example"
ENC_FILE="$ROOT/k8s/encryption/EncryptionConfiguration.yaml"
KIND_CFG="$ROOT/kind/config.yaml"

if [[ ! -f "$ENC_FILE" ]]; then
  KEY="$(head -c 32 /dev/urandom | base64)"
  sed "s|REPLACE_WITH_BASE64_32_BYTE_KEY|$KEY|" "$ENC_EXAMPLE" > "$ENC_FILE"
  echo "Wrote $ENC_FILE with a fresh lab key (gitignored if you add it — keep local)."
  echo "NOTE: commit only the .example file."
fi

# kind extraMounts hostPath must be absolute
ABS_ENC="$(cd "$(dirname "$ENC_FILE")" && pwd)/$(basename "$ENC_FILE")"
TMP_KIND="$(mktemp)"
sed "s|hostPath: ./k8s/encryption/EncryptionConfiguration.yaml|hostPath: $ABS_ENC|" \
  "$KIND_CFG" > "$TMP_KIND"

if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  echo "Cluster $CLUSTER_NAME already exists. Delete first to recreate with encryption:"
  echo "  kind delete cluster --name $CLUSTER_NAME"
  rm -f "$TMP_KIND"
  exit 0
fi

kind create cluster --name "$CLUSTER_NAME" --config "$TMP_KIND"
rm -f "$TMP_KIND"
echo "Cluster ready. Continue TASKS.md §2 (create a Secret, then verify etcd ciphertext)."
