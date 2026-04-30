# kind + Cilium lab

A small multi-node Kubernetes cluster running on Docker, with Cilium as the CNI (eBPF) instead of the default kindnet. Used in W14 and W16.

## Prerequisites

- Docker (in your Linux VM via colima/multipass).
- `kind` (https://kind.sigs.k8s.io/).
- `kubectl`.
- `cilium` CLI (https://docs.cilium.io/en/stable/gettingstarted/k8s-install-default/).
- `helm` (used by some manifests).

```bash
brew install kind kubectl helm
brew install cilium-cli
```

## Bring up the cluster

```bash
cd lab/kind

kind create cluster --config kind-config.yaml --name net-lab

bash cilium-install.sh

cilium status --wait
cilium connectivity test
```

## Manifests

`manifests/` contains progressive examples used in W14:

- `01-frontend-backend.yaml` - two deployments + ClusterIP Services.
- `02-network-policy.yaml` - default-deny + explicit allow frontend->backend.
- `03-coredns-debug.yaml` - a debug pod for DNS exploration.
- `04-ingress.yaml` - ingress-nginx or Cilium Gateway example.

Apply them in order with `kubectl apply -f manifests/01-...` etc.

## Tear-down

```bash
kind delete cluster --name net-lab
```
