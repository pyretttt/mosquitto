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

This repo is intentionally a **scaffold**. Enough works end-to-end that you can
run `docker compose up`, hit the API, and see metrics in Grafana. Everything
marked `TODO(you)` in the code or in `docs/LEARNING_PATH.md` is a deliberate
learning exercise.

| Area | Scaffolded | Your job |
| --- | --- | --- |
| Docker Compose | all 8 services wired, volumes, network | tune memory limits |
| `ml-app` | FastAPI `/predict` + `/health` + `/metrics`, sklearn trainer, MLflow logging | add more features, more tests, model versioning logic |
| MLflow | tracking server w/ SQLite + artifact serving | register model in Model Registry, promote to `Production` stage |
| Prometheus | scrape config + 1 example alert rule | write more alerts, add Alertmanager |
| Grafana | Prometheus datasource, 1 starter API dashboard | build a container-health dashboard (cAdvisor), SLO dashboard |
| GitLab | compose service + runner registration script | register the runner, create project, push code |
| CI pipeline | `.gitlab-ci.yml` with 4 stages (lint/test/train/build) | wire each stage; handle registry auth |
| K8s migration | notes in `docs/K8S_MIGRATION.md` | do the migration |

## Quick start

```bash
cp .env.example .env
docker compose up -d mlflow ml-app prometheus grafana cadvisor node-exporter
# Wait ~20s for ml-app to train + boot.
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
open http://localhost:3000       # Grafana  (admin / admin — change on first login)
open http://localhost:9090       # Prometheus
open http://localhost:5001       # MLflow
```

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
