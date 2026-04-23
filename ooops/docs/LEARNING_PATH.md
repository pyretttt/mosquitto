# Learning path

Do these in order. Each step ends with a concrete **outcome** you can verify.
The scaffolding gets you from 0 → a running stack; the exercises get you from
a running stack → understanding MLOps.

Heads-up on resources: GitLab CE wants ~4 GB RAM on its own. On macOS, open
Docker Desktop → Settings → Resources and give Docker at least **6 GB RAM** and
4 CPUs before you start step 5. If your laptop chokes, skip GitLab and use
[Gitea](https://gitea.io/) instead (drop-in, ~200 MB).

---

## 1. Bring up the stack and confirm the failure modes

```bash
cp .env.example .env
docker compose up -d mlflow ml-app prometheus grafana cadvisor node-exporter
docker compose logs -f ml-app     # watch it (probably) fail to reach mlflow
```

**Expected outcome — read carefully, this is by design:**

- `/predict` *might* work, depending on whether MLflow came up before training ran.
  If training failed on cold start, `docker compose restart ml-app` until it succeeds.
- `curl localhost:8000/health` → 404 (you haven't built it).
- `curl localhost:8000/metrics` → 404 (you haven't built it).
- <http://localhost:9090/targets> shows `ml-app` target **DOWN** (no `/metrics`).
- Grafana starter dashboard's only panel reads **DOWN**.

That's your starting point. The next steps fix it.

## 2. Fix the MLflow start-up race

Open `ml-app/entrypoint.sh`. Read the PROBLEM block at the top, then implement
one of the three options listed there. Verify by:

```bash
docker compose down ml-app mlflow
docker compose up -d mlflow ml-app
docker compose logs ml-app | grep -i train
```

**Outcome:** training succeeds on a cold start, every time.

## 3. Wire /health and /metrics in ml-app

Open `ml-app/TODO.md`. Do the first three items in the "API surface" section:

- `GET /health`
- `Instrumentator()...expose(app)` to mount `/metrics`
- a `Counter` for predictions per class

**Outcome:**

- `curl localhost:8000/health` → 200 with model status.
- `curl localhost:8000/metrics` → Prometheus exposition format.
- <http://localhost:9090/targets> shows `ml-app` **UP**.
- The starter dashboard's `ml-app up` panel turns green.

## 4. Generate traffic and build out the API dashboard

Write a tiny load generator — shell loop or Python — that hits `/predict` with
varied inputs a few times a second for a minute.

Then open `grafana/TODO.md` and add the panels listed under "ML API dashboard":
request rate, p50/p95/p99 latency, error rate, predictions by class.

**Outcome:** the starter dashboard goes from one panel to five, all live.

## 5. Write a real alert rule

Open `prometheus/TODO.md`. Implement at least one alert from the "Alert rules"
section. Reload Prometheus without restarting:

```bash
curl -X POST http://localhost:9090/-/reload
```

**Outcome:** <http://localhost:9090/alerts> shows your rule. Induce the
condition (e.g. slow path with `time.sleep`) and watch it transition
`inactive` → `pending` → `firing`.

## 6. Bring up GitLab and register the runner

```bash
docker compose up -d gitlab
# wait ~5 minutes — first boot takes forever. Tail:
docker compose logs -f gitlab | grep -i 'gitlab Reconfigured'
```

Then:

```bash
docker compose exec gitlab grep 'Password:' /etc/gitlab/initial_root_password
# Login as root at http://localhost:8080 with that password.
# Admin Area -> CI/CD -> Runners -> New instance runner -> copy the token.
docker compose up -d gitlab-runner
# Read scripts/register-runner.sh end-to-end first — it has 4 known issues
# marked TODO(you). Fix at least #1 (token-on-argv) and #3 (idempotency)
# before running it.
./scripts/register-runner.sh <REGISTRATION_TOKEN>
```

**Outcome:** Admin Area → Runners shows one active runner tagged `docker,local`.

## 7. Push the ml-app to GitLab and run the pipeline

1. Create a new project in GitLab (root user is fine).
2. Set it up as a remote:
   ```bash
   cd ml-app
   git init
   git add .
   git commit -m "initial"
   git remote add origin http://localhost:8080/root/ml-app.git
   git push -u origin main
   ```
3. Watch the pipeline run at `http://localhost:8080/root/ml-app/-/pipelines`.

**Outcome:** `lint`, `test`, `train` all green on the first push. `build` will
need the Container Registry enabled — that's the next step.

## 8. Wire the `build` stage to the GitLab Container Registry

1. In the project: Settings → General → Visibility → enable *Container Registry*.
2. Uncomment the `docker login` + `docker push` lines in `.gitlab-ci.yml`.
3. Push. The pipeline will now push the image to the project's registry.

**Outcome:** Packages & Registries → Container Registry shows your image
tagged with the commit SHA.

## 9. Use the Model Registry properly

Right now the API always loads `models:/iris-classifier/None` (= latest
version). Learn the registry workflow:

1. In the MLflow UI, find your model, transition version 1 to **Staging**.
2. Set `ML_MODEL_STAGE=Staging` in `.env` and `docker compose up -d ml-app`.
3. Re-run `python -m src.train` locally to register version 2.
4. Promote version 2 to Staging; demote version 1 to Archived.
5. Restart `ml-app`. `/health` should show the new `model_source`.

**Outcome:** you can flip which model version serves traffic *without*
changing code — only the registry.

## 10. Move training out of the API container

Right now `entrypoint.sh` trains on startup — convenient for demos, bad in
practice (your inference container shouldn't know how to train).

1. Delete the `python -m src.train` line from `entrypoint.sh`.
2. Make the `train` CI job push to the *shared* MLflow server (set
   `MLFLOW_TRACKING_URI` as a GitLab CI/CD Variable pointing to your MLflow).
3. After the CI job trains a new version, the API still needs to pick it up.
   Add an authenticated `POST /admin/reload` route that calls `model.load()`.
4. Add a final `deploy` stage to `.gitlab-ci.yml` that `curl`s `/admin/reload`
   after a successful `train`.

**Outcome:** pushing to GitLab trains a new model and the API hot-reloads it.

## 11. Harden it a bit

Now that the loop is closed, work through whichever per-area `TODO.md` files
interest you most:

- `docker-compose.yml` inline TODOs — resource limits, healthchecks, secrets, network split.
- `prometheus/TODO.md` — Alertmanager, recording rules, blackbox exporter.
- `grafana/TODO.md` — container-health dashboard (cAdvisor), SLO dashboard, alerting.
- `ml-app/TODO.md` — input validation, structured logging, OpenTelemetry tracing.
- `mlflow/README.md` TODOs — Postgres backend, MinIO artifact store.

---

When you've done 1–10 you've covered: CI/CD for ML, experiment tracking,
model registry, service monitoring, alerting, and container health — the
core of MLOps. The rest (feature stores, data versioning, batch scoring,
shadow deploys) builds on this base. Then move to `docs/K8S_MIGRATION.md`.
