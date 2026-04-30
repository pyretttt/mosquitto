#!/usr/bin/env bash
# Install Cilium on the kind cluster.
# Replaces kube-proxy with Cilium's eBPF datapath.

set -euo pipefail

CTX=${1:-kind-net-lab}

cilium install \
  --version=1.16.0 \
  --context "$CTX" \
  --set kubeProxyReplacement=true \
  --set k8sServiceHost=net-lab-control-plane \
  --set k8sServicePort=6443 \
  --set ipam.mode=kubernetes \
  --set hubble.enabled=true \
  --set hubble.relay.enabled=true \
  --set hubble.ui.enabled=true

cilium status --wait

cat <<EOF
ready.

  cilium connectivity test
  cilium hubble ui          # opens browser at hubble UI
  kubectl get pods -A
EOF
