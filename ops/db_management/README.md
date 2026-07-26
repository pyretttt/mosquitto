# db_management

Hands-on lab for **Postgres on Kubernetes**: expose a small FastAPI records API
through **Ingress with TLS termination**, practice **backup/restore**, lock down
east-west traffic with **NetworkPolicies**, and (optionally) run backups under a
dedicated **ServiceAccount**. Extra track: tune **ingress-nginx** ConfigMap /
annotations.

> This is a **scaffold**. Most pieces are intentionally left as exercises.
> Look for `TODO(you)` markers and the checklists in `TASKS.md`.

## Learning outcomes

- Deploy Postgres + a FastAPI app on kind with Helm and reproducible mise tasks
- Terminate TLS at ingress-nginx using a self-signed cert and verify with curl
- Backup and restore Postgres (`pg_dump` / `pg_restore`) and move that into a Job
- Apply default-deny NetworkPolicies and reopen only API↔DB and Ingress→API
- Wire a least-privilege ServiceAccount + Role for backup automation
- Change nginx controller ConfigMap and Ingress annotations and prove the effect

## Components

| Concern | Path | Backed by |
| ------- | ---- | --------- |
| Tooling / tasks | `mise.toml` | mise |
| Concepts | `docs/CONCEPTS.md` | markdown |
| Records API | `app/` | Python FastAPI + psycopg |
| API chart | `charts/records-api/` | Helm |
| Postgres values | `charts/postgresql/` | Bitnami PostgreSQL (OCI) |
| kind cluster | `kind/config.yaml` | kind + ingress ports |
| Ingress / TLS | `k8s/ingress.yaml.example`, `certs/` | ingress-nginx + openssl |
| NetworkPolicies | `k8s/networkpolicy/` | NetworkPolicy API |
| Backup RBAC / Job | `k8s/rbac/`, `k8s/backup/` | batch/v1 + RBAC |
| nginx knobs | `k8s/nginx/` | ingress-nginx ConfigMap |
| Scripts | `scripts/` | bash |

## Layout

```
db_management/
├── README.md
├── TASKS.md
├── mise.toml
├── .env.example
├── docs/CONCEPTS.md
├── app/
├── charts/
│   ├── records-api/
│   └── postgresql/values-local.yaml
├── kind/config.yaml
├── certs/                 # generated TLS (gitignored)
├── k8s/
│   ├── namespace.yaml
│   ├── ingress.yaml.example
│   ├── networkpolicy/
│   ├── rbac/
│   ├── backup/
│   └── nginx/
└── scripts/
```

## Prerequisites

- Docker Desktop (or another Docker engine) with enough RAM for kind
- [mise](https://mise.jdx.dev/)
- `openssl` (`brew install openssl` / `apt install openssl`)
- Ability to bind host ports **80** and **443** (or adjust `kind/config.yaml`)

## Quick start

```bash
cd ops/db_management
cp .env.example .env          # set POSTGRES_PASSWORD
mise install
mise run verify-tools

mise run cluster-up
mise run ns
mise run ingress-install
mise run deploy-postgres
mise run build-load
mise run deploy-app

# Sanity without Ingress yet:
mise run port-forward
# other terminal:
curl -s localhost:8080/health
curl -s localhost:8080/records
```

Then continue with TLS, backup, NetworkPolicies, and nginx in `TASKS.md`.

## Where to work

Start with `TASKS.md`. Pair each section with `docs/CONCEPTS.md`.
