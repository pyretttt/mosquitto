#!/usr/bin/env bash
# Peek at a Secret's etcd value inside the kind control-plane.
# With encryption-at-rest, you should NOT see plaintext / plain base64 of the secret data.
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-secrets-lab}"
NS="${NS_APP:-demo}"
SECRET_NAME="${1:-demo-app-secret}"

NODE="${CLUSTER_NAME}-control-plane"
ETCD_POD_PREFIX="" # kind runs etcd as a static pod on the control-plane

echo "Looking up Secret $NS/$SECRET_NAME in etcd on $NODE ..."
# kind mounts etcd certs under /etc/kubernetes/pki/etcd
docker exec "$NODE" sh -c "
  ETCDCTL_API=3 etcdctl \
    --endpoints=https://127.0.0.1:2379 \
    --cacert=/etc/kubernetes/pki/etcd/ca.crt \
    --cert=/etc/kubernetes/pki/etcd/server.crt \
    --key=/etc/kubernetes/pki/etcd/server.key \
    get /registry/secrets/${NS}/${SECRET_NAME} 2>/dev/null | head -c 400 || true
"
echo
echo "If encryption-at-rest is active, the blob should look opaque (k8s:enc:aescbc:v1:key1:...), not readable YAML."
echo "TODO(you): compare before/after enabling EncryptionConfiguration — TASKS.md §2"
