# vault_proj

A small beginner lab for **HashiCorp Vault**: learn what problem secrets managers solve,
practice the CLI on your laptop, then wire a tiny demo app on Kubernetes so it reads
backend secrets (and optional TLS certs) from Vault instead of baking them into images
or plain Kubernetes Secrets.

> This is a **scaffold**. Most pieces are intentionally left as exercises.
> Look for `TODO(you)` markers and the checklists in `TASKS.md`.

## What problem does Vault solve?

Apps and operators need credentials everywhere: DB passwords, API keys, TLS certs,
cloud tokens. The naive approaches fail in different ways:

| Approach | Failure mode |
| -------- | ------------ |
| Hardcode in source / images | Leaks in git history, registries, and every clone |
| Env vars / ConfigMaps | Visible to anyone with pod/exec or cluster RBAC |
| Plain Kubernetes Secrets | Base64, not encryption-at-rest by default; no rotation story |
| Shared “ops password doc” | No audit trail, no expiry, no least privilege |

**Vault** is a centralized secrets engine: authenticate → get a short-lived credential
(or a dynamic one Vault generates) → use it → renew or let it expire → audit who did what.
It also handles encryption-as-a-service, PKI (issue your own TLS without a paid CA),
and identity-based access (Kubernetes ServiceAccounts, AppRole, tokens, …).

### Mental model (how Vault works)

```
  Client (CLI / app / CI)
       │  1. Authenticate (token, K8s SA, AppRole, …)
       ▼
  Vault ── Auth method ──► Identity + policies (what paths you may touch)
       │
       ├── Secrets engines (KV, database, PKI, …)
       ├── Leases (TTL on dynamic secrets / tokens)
       └── Audit devices (who read what, when)
```

Important lifecycle ideas you will practice here:

- **Seal / unseal** — Vault starts sealed; master key shares unseal it (dev mode auto-unseals).
- **Tokens** — proof of identity after auth; inherit policies; can be renewable / revocable.
- **Leases** — Vault-issued credentials often have a lease id; **renew** or **revoke**.
- **Namespaces** (Enterprise / HCP feature conceptually) — isolation boundaries; lock/unlock.
- **Rotation / expiry** — short TTLs + renew/rotate beats “password forever”.

Read `docs/CONCEPTS.md` for the theory track; do `TASKS.md` for hands-on.

## Learning outcomes

- Explain why a secrets manager beats env vars / plain K8s Secrets
- Run Vault locally and use the CLI: write/read KV, tokens, leases
- Understand seal/unseal, auth methods, policies, and token lifecycle
- Deploy Vault + a demo backend on a local Kubernetes cluster (kind/k3d)
- Inject or fetch app secrets from Vault; issue TLS via Vault PKI (self-signed / private CA)
- Practice secret TTL, renew, revoke, and a simple rotation workflow

## Components

| Concern | Path | Backed by |
| ------- | ---- | --------- |
| Tooling / tasks | `mise.toml` | mise |
| Concepts (theory) | `docs/CONCEPTS.md` | markdown |
| Local CLI lab | `scripts/local-*.sh`, `local/` | Vault `-dev` or file storage |
| Demo web API | `app/` | Python FastAPI |
| Vault on K8s values | `charts/vault-values/` | HashiCorp Vault Helm chart |
| Demo app chart | `charts/demo-app/` | Helm |
| Bootstrap helpers | `k8s/`, `scripts/` | kubectl / helm |

## Layout

```
vault_proj/
├── README.md
├── TASKS.md
├── mise.toml
├── .env.example
├── docs/
│   └── CONCEPTS.md
├── local/
│   └── vault-config.hcl.example
├── scripts/
│   ├── local-dev.sh
│   ├── local-kv-demo.sh
│   └── k8s-bootstrap.sh
├── app/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── charts/
│   ├── vault-values/
│   │   ├── values-local.yaml
│   │   └── README.md
│   └── demo-app/
│       ├── Chart.yaml
│       ├── values.yaml
│       ├── values-local.yaml
│       └── templates/
└── k8s/
    └── namespace.yaml
```

## Prerequisites

- [mise](https://mise.jdx.dev/) installed
- Docker (for kind/k3d and building the demo image)
- Curiosity; no prior Vault experience required

## Quick start

```bash
cd ops/vault_proj

# Install pinned tools (vault, kubectl, helm, kind/k3d, …)
mise install

# --- Track A: local CLI only (no cluster) ---
mise run vault-dev          # Vault in -dev mode on :8200
# in another terminal:
mise run local-kv-demo      # write/read a secret, show token basics

# --- Track B: Kubernetes demo ---
mise run cluster-k3d        # or: mise run cluster-kind
mise run vault-install      # Helm install Vault (dev/server per values)
mise run app-build          # build + load demo image into the cluster
mise run app-install        # deploy demo-app chart
```

Open `TASKS.md` and work section 0 → … in order.

## Where to work

Start with **`TASKS.md`**. Theory that pairs with each section lives in **`docs/CONCEPTS.md`**.
