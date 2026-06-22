# mlopster

A Kubernetes-native MLOps platform scaffold, deployed entirely with Helm.

It is built to run on a **local cluster** (kind / k3d / minikube / Docker Desktop)
but every chart is written to be **portable to a real cluster** — the
environment-specific bits (storage class, ingress, resource sizing, replica
counts, secrets) live in `values-local.yaml` overlays, while the base
`values.yaml` holds the portable, production-leaning defaults.

> This is a **scaffold**. Most pieces are intentionally left as exercises.
> Look for `TODO(you)` markers in code and the per-component checklists in
> `TASKS.md`. Nothing here is "finished" — it is a skeleton with working
> wiring and clear seams.

## Components

| Concern            | Chart / dir            | Backed by                                   |
| ------------------ | ---------------------- | ------------------------------------------- |
| Metrics & alerting | `charts/monitoring`    | `kube-prometheus-stack` (Prometheus Operator, Alertmanager, Grafana) |
| Dashboards         | `grafana/`             | Grafana (provisioned from files)            |
| Object storage (S3)| `charts/storage`       | MinIO (S3-compatible)                       |
| Orchestration      | `charts/orchestration` | Apache Airflow (training DAGs)              |
| Experiment tracking| `charts/orchestration` | MLflow — **reserved, disabled for now**     |
| Model serving      | `charts/model-server`  | FastAPI + Prometheus instrumentation        |
| CI                 | `.gitlab-ci.yml`, `ci/`| GitLab CI (free tier shared runners)        |

## Layout

```
mlopster/
├── charts/
│   ├── monitoring/      # kube-prometheus-stack umbrella + app CRDs (ServiceMonitor, PrometheusRule)
│   ├── storage/         # MinIO (S3) umbrella + bucket bootstrap job
│   ├── orchestration/   # Apache Airflow umbrella (+ MLflow reserved)
│   └── model-server/    # our FastAPI inference service
├── model-server/        # source for the FastAPI app (built into an image by CI)
├── airflow/             # DAGs + airflow-local python deps
├── grafana/             # provisioning: datasources, dashboards, alerting
├── ci/                  # reusable GitLab CI templates
├── docs/                # ARCHITECTURE.md, ADMIN_TASKS.md (certs/network/security)
├── mise.toml            # tool versions + bootstrap tasks (`mise tasks`)
└── TASKS.md             # the master task list (start here)
```

## Quick start (local)

Prereqs: [`mise`](https://mise.jdx.dev) and `docker`. `mise install` provisions
`helm`, `kubectl`, `kind`, and `k3d` from `mise.toml`; run `mise tasks` to see
everything available.

```bash
# 0. install the pinned tools (helm/kubectl/kind/k3d)
mise install

# 1. create a local cluster (pick your tool)
mise run cluster-kind        # or: mise run cluster-k3d

# 2. add helm repos and build chart dependencies
mise run deps

# 3. install everything into the local cluster
mise run up-local

# 4. port-forward the UIs
mise run pf-grafana          # http://localhost:3000
mise run pf-prometheus       # http://localhost:9090
mise run pf-model            # http://localhost:8000/docs
```

Going to a real cluster: drop the `-local` overlays, supply a real
`storageClass`/`ingress`, and move secrets out of values into a secret manager
(see `docs/ADMIN_TASKS.md`).

## Where to start

Read `TASKS.md`. It is ordered roughly easiest → hardest and grouped by
component. The admin/hardening track (certs, network policy, RBAC, secrets) is
deliberately kept **separate** at the bottom — do the main tracks first.
