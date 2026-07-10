# Vault Helm values for this lab

This directory is **not** a chart — it holds values overlays for the upstream
[`hashicorp/vault`](https://developer.hashicorp.com/vault/docs/platform/k8s/helm)
chart.

```bash
mise run vault-repos
mise run vault-install
# → helm upgrade --install vault hashicorp/vault -n vault -f values-local.yaml
```

| File | Purpose |
| ---- | ------- |
| `values-local.yaml` | Disposable local cluster (dev server on by default) |

After install, get the root token (dev mode often uses `root` as configured) and
port-forward to talk to Vault from your laptop with the `vault` CLI.
