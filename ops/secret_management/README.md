# secret_management

A hands-on lab for **Kubernetes secret management**: what ConfigMaps and Secrets
actually are, why base64 is not encryption, how encryption-at-rest works, and
how teams keep credentials out of plaintext git with **SOPS**, **Sealed Secrets**,
and **External Secrets**.

> This is a **scaffold**. Most pieces are intentionally left as exercises.
> Look for `TODO(you)` markers and the checklists in `TASKS.md`.

Vault is out of scope here — use `ops/vault_proj` for that track.

## Learning outcomes

- Explain ConfigMap vs Secret and why base64 ≠ encryption
- Enable and verify Kubernetes encryption-at-rest for Secrets
- Encrypt manifests with SOPS + age and apply them safely
- Install Sealed Secrets and seal/unseal via GitOps-friendly CRs
- Install External Secrets Operator and sync from a lab store (Fake / Kubernetes)
- Point a small demo app at each Secret source and verify without leaking values

## Components

| Concern | Path | Backed by |
| ------- | ---- | --------- |
| Tooling / tasks | `mise.toml` | mise |
| Concepts | `docs/CONCEPTS.md` | markdown |
| Demo API | `app/` | Python FastAPI |
| Plain CM / Secret examples | `k8s/` | kubectl |
| Encryption-at-rest | `k8s/encryption/`, `kind/` | kind + kube-apiserver |
| SOPS | `.sops.yaml`, `secrets/` | sops + age |
| Sealed Secrets | `sealed-secrets/` | Bitnami sealed-secrets |
| External Secrets | `external-secrets/` | ESO (Fake / K8s provider) |
| Demo chart | `charts/demo-app/` | Helm |

## Layout

```
secret_management/
├── README.md
├── TASKS.md
├── mise.toml
├── .env.example
├── .sops.yaml
├── docs/CONCEPTS.md
├── app/
├── k8s/
│   ├── namespace.yaml
│   ├── 01-configmap.yaml
│   ├── 02-secret.yaml.example
│   └── encryption/
├── kind/config.yaml
├── secrets/                 # SOPS plaintext (gitignored) + ciphertext
├── sealed-secrets/
├── external-secrets/
├── charts/demo-app/
└── scripts/
```

## Prerequisites

- Docker
- [mise](https://mise.jdx.dev/)
- Optional: `GITHUB_TOKEN` in the environment if `mise install` hits GitHub rate limits (sops / age / kubeseal)
- ~6–9 hours across 2–3 days

## Quick start

```bash
cd ops/secret_management
cp .env.example .env
mise install
mise run cluster-kind
mise run ns
mise run configmap
# create a Secret, then:
mise run secret-create   # needs DEMO_API_TOKEN in env
mise run build-load
mise run deploy-app
mise run port-forward
# curl http://127.0.0.1:8080/config
# curl http://127.0.0.1:8080/secret-status
```

## Where to work

Start with `TASKS.md`. Skim `docs/CONCEPTS.md` alongside each section.
