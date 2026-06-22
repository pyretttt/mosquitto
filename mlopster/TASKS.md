# mlopster — Master Task List

Start here. Tasks are grouped by component and ordered roughly easiest →
hardest within each group. Inline `TODO(you)` markers in the code point back to
these items. Tick them off as you go.

Legend: `[ ]` todo · `[~]` partially scaffolded · `[x]` done

The **Admin & Hardening** track at the bottom (certs, networking, RBAC,
secrets) is intentionally separate — do the main tracks first, then harden.

---

## 0. Bootstrap

- [~] Create a local cluster (`mise run cluster-kind` or `mise run cluster-k3d`).
- [~] Add helm repos + build deps (`mise run repos && mise run deps`).
- [ ] Decide your local storage class. kind/minikube → `standard`,
      k3d/k3s → `local-path`. Set it once in each `values-local.yaml`.
- [ ] `mise run up-local` and confirm every pod reaches `Running`.

---

## 1. Prometheus / monitoring  (`charts/monitoring`)

We use the `kube-prometheus-stack` chart from the prometheus-community repo.
It bundles the **Prometheus Operator**, which is the key idea here: you don't
hand-edit `prometheus.yml`. Instead you create **CRDs** and the operator
reconciles them into the running Prometheus config.

### How config & alerts are stored (the thing to actually understand)

There are several ways to feed Prometheus its scrape config and alert rules.
The scaffold demonstrates the CRD-first approach but you should know all of them:

1. **CRDs (recommended, what we do):**
   - `ServiceMonitor` / `PodMonitor` → scrape targets. See
     `charts/monitoring/templates/servicemonitor-model-server.yaml`.
   - `PrometheusRule` → alerting + recording rules. See
     `charts/monitoring/templates/prometheusrule-model-server.yaml`.
   - `AlertmanagerConfig` → routing/receivers as a namespaced CRD (alternative
     to the big global `alertmanager.config` blob). See the stub in
     `charts/monitoring/templates/alertmanagerconfig-example.yaml`.
   - The operator only picks up CRDs whose **labels match the selectors** on
     the `Prometheus`/`Alertmanager` resources. With kube-prometheus-stack the
     default selector is `release: <helm-release-name>`. That's why every CRD
     here carries `labels.release: monitoring`.
- [ ] **Task:** break the label on purpose, reload, and watch the target/rule
      disappear from the Prometheus UI. Understand `*SelectorNilUsesHelmValues`.

2. **Inline via Helm values:**
   - `prometheus.prometheusSpec.additionalScrapeConfigs` for raw scrape jobs.
   - `additionalPrometheusRulesMap` to define rules straight in values.
   - `alertmanager.config` for the whole Alertmanager config (see
     `charts/monitoring/values.yaml`).
- [ ] **Task:** add one `additionalScrapeConfigs` job (e.g. blackbox probe of
      the model server `/health`) and compare the ergonomics vs a ServiceMonitor.

3. **`additionalScrapeConfigsSecret`** — keep raw scrape config (with creds) in
   a Secret instead of values. Note when you'd prefer this.
- [ ] **Task:** write a one-paragraph note in `docs/ARCHITECTURE.md` on when to
      use CRDs vs inline values vs secret-backed config.

### Storage, retention, HA

- [~] Persistent storage for Prometheus TSDB + Alertmanager (volumeClaimTemplate
      is scaffolded; pick sizes per env).
- [ ] Set `retention` / `retentionSize` sensibly for local vs prod.
- [ ] **Task:** read about Thanos / remote-write for long-term storage and
      note the migration path (don't implement yet).
- [ ] **Task:** add a `ServiceMonitor` for MinIO and one for Airflow's
      statsd/metrics exporter so the whole platform is observable.

### Alerting (Prometheus/Alertmanager side)

- [~] `PrometheusRule` with `ModelServerDown`, `ModelServerHighErrorRate`,
      `ModelServerHighLatency` is scaffolded — verify the exprs match the
      metric names your app actually exposes.
- [ ] Wire a real Alertmanager receiver (Slack/email/webhook) in
      `charts/monitoring/values.yaml` → `alertmanager.config`. Keep the secret
      bits out of git (see Admin track).

---

## 2. Grafana  (`charts/monitoring` values + `grafana/`)

Grafana ships **inside** kube-prometheus-stack. We configure it via the
`grafana:` block in `charts/monitoring/values.yaml` and provision dashboards /
datasources / alerting from files in `grafana/`.

### Persistence

- [~] `grafana.persistence.enabled: true` with a PVC (scaffolded).
- [ ] Decide: PVC (simple) vs an external DB (`grafana.ini` → `[database]`,
      Postgres) for HA. Note the tradeoff. For >1 replica you need the DB.

### Security

- [ ] Move `adminPassword` out of values into a Secret
      (`grafana.admin.existingSecret`). Generate it; never commit it.
- [ ] Set `grafana.ini` security flags: `cookie_secure`, `disable_gravatar`,
      `[users] allow_sign_up = false`, and a real `root_url`.
- [ ] Decide auth: keep local admin for now; note OAuth/OIDC for "real cluster".
- [ ] Mark provisioned datasources `editable: false` so the UI can't drift
      from git.

### Provisioning (files in `grafana/`)

- [~] Datasource → Prometheus (`grafana/provisioning/datasources/`).
- [~] Dashboard provider + a starter ML-API dashboard
      (`grafana/provisioning/dashboards/`, `grafana/dashboards/`).
- [ ] Build panels in the UI, then export JSON into `grafana/dashboards/` so
      it's reproducible. Suggested panels: request rate by handler+status,
      p50/p95/p99 latency, error rate, predictions/sec by class, model-only
      latency p95.
- [ ] Decide how dashboards get into Grafana: the sidecar (ConfigMap discovery,
      what this scaffold uses via `grafana.sidecar.dashboards`) vs baking files
      into the image vs `dashboardProviders` paths. Note pros/cons.

---

## 3. Alerting through Grafana  (`grafana/provisioning/alerting/`)

This is **Grafana unified alerting**, separate from Prometheus/Alertmanager
rules. Same data, different engine — know when to use each.

- [~] Provision an alert rule mirroring `ModelServerDown` via files
      (`grafana/provisioning/alerting/rules.yaml`).
- [~] Provision a contact point (`contact-points.yaml`) — start with a
      webhook.site URL.
- [~] Provision a notification policy routing `severity=critical`
      (`policies.yaml`).
- [ ] **Task:** trigger it (scale the model server to 0) and confirm the
      contact point fires.
- [ ] **Task:** write a note: when would you alert from Prometheus rules +
      Alertmanager vs from Grafana? (Hint: data sources, multi-DC, UI vs GitOps.)

---

## 4. GitLab CI  (`.gitlab-ci.yml`, `ci/`)

Targets the **free tier** with shared runners. Stages: lint → test → build →
(optional) deploy. Build uses Kaniko (no privileged Docker needed on shared
runners).

- [~] `lint` stage: ruff + helm lint (scaffolded in `ci/lint.gitlab-ci.yml`).
- [~] `test` stage: pytest for the model server (`ci/test.gitlab-ci.yml`).
- [~] `build` stage: Kaniko builds + pushes the model-server image to the
      GitLab Container Registry (`ci/build.gitlab-ci.yml`).
- [ ] Set `CI_REGISTRY_*` usage and tag images with `$CI_COMMIT_SHORT_SHA`.
- [ ] Add a `helm template`/`helm lint` job so chart errors fail the pipeline.
- [ ] (Stretch) add a `deploy` stage gated on `main` that runs `helm upgrade`
      against a cluster reachable from the runner (or just `--dry-run`).
- [ ] Cache pip between runs (`.cache/pip`) to stay fast on free runners.
- [ ] Add a `train` stage placeholder (kept manual) for when MLflow lands.

---

## 5. MLflow — RESERVED (do not enable yet)

We are **not** adding MLflow now, but the seams exist:

- [~] `charts/orchestration/Chart.yaml` has the MLflow dependency commented out.
- [~] `charts/orchestration/values.yaml` has a commented `mlflow:` block.
- [~] The model server reads `MLFLOW_TRACKING_URI` (unused until enabled) and
      `model.py` has a `# TODO(you): load from MLflow registry` seam.
- [ ] **Later:** enable the dependency, point Airflow training DAGs at the
      tracking server, and switch `model.py` to load from the MLflow registry
      instead of S3.

---

## 6. Model server (FastAPI)  (`model-server/`, `charts/model-server`)

- [x] `POST /predict` — validate input, return prediction.
- [~] `GET /health` — liveness/readiness, reports whether the model is loaded.
- [~] `GET /metrics` — Prometheus exposition via instrumentator.
- [~] Custom `Counter` (predictions by class) and `Histogram` (model latency).
- [~] Load the model artifact from **S3 (MinIO)** on startup, with a safe
      fallback so the service still boots when no artifact exists yet.
- [ ] Add an authenticated `POST /admin/reload-model` to pick up new artifacts
      without a restart.
- [ ] Make readiness fail (503) until the model is loaded so k8s doesn't send
      traffic to a model-less pod. Confirm the `readinessProbe` in the chart
      uses `/health`.
- [ ] Add an HPA (`charts/model-server/templates/hpa.yaml` is stubbed) and
      load-test to see it scale.
- [ ] Flesh out tests in `model-server/tests/` (health 200/503, metrics format,
      predict validation 422).

---

## 7. Apache Airflow (training)  (`charts/orchestration`, `airflow/`)

- [~] Airflow installed via the official chart (umbrella dependency).
- [~] A `train_model` DAG scaffold in `airflow/dags/` that: pulls data from S3,
      trains a model, writes the artifact back to S3.
- [ ] Decide how DAGs reach Airflow: git-sync (recommended, scaffolded in
      values) vs baking into the image. Wire git-sync to this repo's
      `airflow/dags/` path.
- [ ] Give Airflow tasks S3 credentials via an Airflow Connection / k8s Secret
      (not hardcoded).
- [ ] Choose an executor: `KubernetesExecutor` (scaffolded) so each task runs
      in its own pod — good fit for training jobs.
- [ ] **Later (MLflow):** log params/metrics/model to MLflow from the DAG and
      register the model.

---

## 8. S3 / object storage  (`charts/storage`)

- [~] MinIO installed (S3-compatible). Console + API exposed.
- [~] A post-install `Job` creates the buckets (`models`, `data`).
- [ ] Put MinIO credentials in a Secret; reference it from the model server and
      Airflow (single source of truth).
- [ ] **Task:** upload a sample dataset to the `data` bucket (mc or the console)
      so the training DAG has something to read.
- [ ] **Real cluster:** swap MinIO for actual S3 by only changing endpoint +
      credentials in values — the app uses the S3 API, not MinIO specifics.

---

## Admin & Hardening (separate track — do after the main tracks)

See `docs/ADMIN_TASKS.md` for the detailed checklist. Summary:

- [ ] **TLS/Certificates:** install `cert-manager`, issue certs for Grafana /
      Airflow / model-server ingress; internal mTLS options.
- [ ] **Ingress:** install an ingress controller (nginx), put hostnames +
      TLS in front of the UIs (currently port-forward only locally).
- [ ] **NetworkPolicy:** default-deny per namespace, then allow only the flows
      that are needed (app→MinIO, Prometheus→targets, etc.).
- [ ] **Secrets management:** move all passwords/keys out of values into
      Sealed Secrets / External Secrets / SOPS.
- [ ] **RBAC & ServiceAccounts:** least-privilege SAs per component; no default
      SA token automounting where not needed.
- [ ] **Pod security:** `securityContext` (non-root, read-only rootfs, drop
      caps), Pod Security Admission labels per namespace.
- [ ] **Resource governance:** ResourceQuotas + LimitRanges per namespace;
      PodDisruptionBudgets for stateful bits.
- [ ] **Backups:** snapshot Prometheus/Grafana/MinIO PVCs; document restore.
