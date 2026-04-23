# Local MLOps Platform

A self-contained, local MLOps stack for learning. You run it all on your laptop
with `docker compose`, push code to a local GitLab, CI trains a model, the
model is served by FastAPI, metrics flow to Prometheus, and Grafana shows
dashboards.

```
            ┌────────────┐   push     ┌────────────────────┐
            │  You (git) │ ─────────► │  GitLab (8080)     │
            └────────────┘            │   + Runner         │
                                      └─────────┬──────────┘
                                                │ triggers pipeline
                                                ▼
                                      ┌────────────────────┐
                                      │  ml-app (8000)     │
                                      │  FastAPI + sklearn │──► MLflow (5001)
                                      │  /metrics exposed  │
                                      └─────────┬──────────┘
                                                │ scrape
                                                ▼
                      ┌──────────────────┐   ┌─────────────────┐
                      │ Prometheus (9090)│──►│ Grafana (3000)  │
                      └──────────────────┘   └─────────────────┘
                              ▲
                              │ scrape
                  ┌───────────┴───────────┐
                  │                       │
          ┌──────────────┐       ┌──────────────┐
          │ node-exporter│       │   cAdvisor   │
          │   (9100)     │       │   (8081)     │
          └──────────────┘       └──────────────┘
```

## What's here vs. what you'll build

This repo is intentionally a **thin scaffold**. The container plumbing works,
the ML model trains and serves predictions, but most of the *interesting*
parts are deliberately missing. You'll find them as `TODO(you)` comments
inline and as per-area `TODO.md` files:

- `ml-app/TODO.md`
- `grafana/TODO.md`
- `prometheus/TODO.md`
- inline `TODO(you)` in `docker-compose.yml`, `ml-app/entrypoint.sh`, `scripts/register-runner.sh`
- ordered learning sequence in `docs/LEARNING_PATH.md`

| Area | Scaffolded | Your job (highlights) |
| --- | --- | --- |
| Docker Compose | services wired, volumes, network | resource limits, healthchecks, secrets, network split, log rotation |
| `ml-app` | `/predict`, sklearn trainer, MLflow logging, model registry plumbing | wire `/health`, wire `/metrics`, add custom counters & histograms, fix the MLflow start-up race |
| MLflow | tracking server w/ SQLite + artifact serving | swap to Postgres + MinIO, model registry promotion workflow |
| Prometheus | 4 scrape jobs + 1 example alert | write real alerts, add recording rules, add Alertmanager + blackbox exporter |
| Grafana | datasource provisioning + 1-panel starter dashboard | build the API dashboard, build a cAdvisor dashboard, set up alerting |
| GitLab + Runner | compose services + a rough registration script | fix 4 issues in the script, register, push code, wire the pipeline |
| CI pipeline | `.gitlab-ci.yml` skeleton with 4 stages | actually wire each stage, container registry auth, deploy stage |
| K8s migration | notes in `docs/K8S_MIGRATION.md` | do the migration |

## Quick start

```bash
cp .env.example .env
docker compose up -d mlflow ml-app prometheus grafana cadvisor node-exporter
# ml-app may fail to train on first try — that's the MLflow start-up race
# documented in ml-app/entrypoint.sh and is your first exercise to fix.
# Workaround until you fix it: `docker compose restart ml-app`.
curl -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
open http://localhost:3000       # Grafana  (admin / admin — change on first login)
open http://localhost:9090       # Prometheus  (ml-app target will be DOWN until you wire /metrics)
open http://localhost:5001       # MLflow
```

Expected initial state:

- `/predict` works → proves the model trained, was registered in MLflow, and loaded into the API.
- `/health` returns **404** → you haven't built it yet (`ml-app/TODO.md`).
- `/metrics` returns **404** → you haven't wired the instrumentator yet.
- Prometheus shows `ml-app` target as **DOWN** → consequence of the above.
- Grafana starter dashboard shows `ml-app up` as **DOWN** → same root cause.

Fixing that chain is the first real exercise. See `docs/LEARNING_PATH.md`.

Start GitLab separately (it's heavy — see `docs/LEARNING_PATH.md` step 5):

```bash
docker compose up -d gitlab gitlab-runner
# First boot takes ~5 minutes. Then:
open http://localhost:8080
```

## Next

Read [`docs/LEARNING_PATH.md`](docs/LEARNING_PATH.md) for the ordered set of
exercises. Read [`docs/K8S_MIGRATION.md`](docs/K8S_MIGRATION.md) before you
start the k8s migration next week.
