# Architecture

```
                         ┌─────────────────────────────────────────┐
                         │              Kubernetes cluster          │
                         │                                          │
   developer ──git──►  GitLab CI ──build/push──► Container Registry │
                         │                              │           │
                         │   ns: monitoring             │ pull      │
                         │   ┌───────────────────┐      ▼           │
                         │   │ kube-prometheus    │  ns: model-server│
                         │   │  - Prometheus      │  ┌────────────┐ │
                         │   │  - Alertmanager    │◄─┤ FastAPI    │ │
                         │   │  - Grafana         │  │ /metrics   │ │
                         │   └───────────────────┘  │ /predict   │ │
                         │      ▲ ServiceMonitor     └─────┬──────┘ │
                         │      │ PrometheusRule           │ load   │
                         │      │                          ▼        │
                         │   ns: orchestration        ns: storage   │
                         │   ┌───────────────┐        ┌──────────┐  │
                         │   │ Airflow       │──train─►│ MinIO    │  │
                         │   │ (Kubernetes   │  upload │ (S3)     │  │
                         │   │  Executor)    │◄─data───┤ buckets: │  │
                         │   └───────────────┘         │ data,    │  │
                         │                             │ models   │  │
                         │   [MLflow — reserved]       └──────────┘  │
                         └─────────────────────────────────────────┘
```

## Namespaces

| Namespace      | Release        | Contents |
| -------------- | -------------- | -------- |
| `monitoring`   | `monitoring`   | Prometheus Operator, Prometheus, Alertmanager, Grafana, app CRDs |
| `storage`      | `storage`      | MinIO + buckets `data`, `models` |
| `orchestration`| `orchestration`| Airflow (+ reserved MLflow) |
| `model-server` | `model-server` | FastAPI inference service |

## Data / control flow

1. **Train** — Airflow `train_iris_model` reads from `data` bucket, trains, and
   uploads `models/model.joblib` to MinIO.
2. **Serve** — the FastAPI pod downloads that artifact on startup (and on
   `POST /admin/reload-model`) and serves `/predict`.
3. **Observe** — the model server exposes `/metrics`; the Prometheus Operator
   scrapes it via a `ServiceMonitor`; `PrometheusRule` defines alerts; Grafana
   visualizes and (separately) alerts.
4. **Ship** — GitLab CI lints, tests, and builds the image; deploy is a manual
   helm upgrade (or GitOps later).

## Why these choices

- **kube-prometheus-stack + Operator:** config and alerts are *CRDs in git*,
  reconciled by the operator. No hand-edited `prometheus.yml`. Three ways to
  feed it config, by preference:
  - **CRDs** (`ServiceMonitor`/`PodMonitor`, `PrometheusRule`,
    `AlertmanagerConfig`) — declarative, namespaced, GitOps-friendly. Default.
  - **Helm values** (`additionalScrapeConfigs`, `additionalPrometheusRulesMap`,
    `alertmanager.config`) — good for static, central config.
  - **Secret-backed** (`additionalScrapeConfigsSecret`) — when scrape config
    contains credentials.
  The operator only adopts CRDs whose labels match its selectors
  (`release: monitoring` here) — that label is the single most common reason a
  target/rule "doesn't show up".
- **Helm umbrella charts** wrap upstream charts so we own the values in git and
  layer our own CRDs alongside.
- **values.yaml + values-local.yaml overlays** keep the base portable
  (production-leaning defaults, no env specifics) while the `-local` overlay
  carries the local cluster's storage class, smaller resources, and disabled
  control-plane scrape jobs.
- **MinIO for S3** so the app talks the S3 API; moving to real S3 is an
  endpoint + credentials change, no code change.
- **Airflow with KubernetesExecutor** so each training task is its own pod with
  its own resources — natural fit for ML jobs.

## Portability: local → real cluster

| Concern | Local | Real cluster |
| --- | --- | --- |
| Storage class | `standard` / `local-path` in `-local` overlays | your CSI class in base values |
| Ingress/TLS | off; port-forward | nginx + cert-manager (see ADMIN_TASKS) |
| Secrets | plain values (dev only) | External/Sealed Secrets |
| Object storage | MinIO | managed S3 (endpoint + creds only) |
| Image registry | local registry / `kind load` | GitLab registry |
| Resources | tiny requests | sized per node pool |

## Reserved seams (not built yet)

- **MLflow** — dependency commented in `charts/orchestration`; the model server
  already reads `MLFLOW_TRACKING_URI` and `model.py` has a load-from-registry
  TODO.
- **Loki/Tempo** — note in TASKS for logs/traces; Grafana datasource can be
  added later.
- **Thanos / remote-write** — long-term metric storage; noted in TASKS §1.
