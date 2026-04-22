# Learning path

Do these in order. Each step ends with a concrete **outcome** you can verify.
The scaffolding gets you from 0 → a running stack; the exercises get you from
a running stack → understanding MLOps.

Heads-up on resources: GitLab CE wants ~4 GB RAM on its own. On macOS, open
Docker Desktop → Settings → Resources and give Docker at least **6 GB RAM** and
4 CPUs before you start step 5. If your laptop chokes, skip GitLab and use
[Gitea](https://gitea.io/) instead (drop-in, ~200 MB).

---

## 1. Bring up the observability + ML stack

```bash
cp .env.example .env
docker compose up -d mlflow ml-app prometheus grafana cadvisor node-exporter
docker compose logs -f ml-app     # watch it train + start
```

**Outcome:**

- `curl http://localhost:8000/health` → `{"status":"ok","model_loaded":true,...}`
- `curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"features":[5.1,3.5,1.4,0.2]}'` → `{"predicted_class":0,...}`
- <http://localhost:5001> shows one run under experiment `iris-classifier`.
- <http://localhost:9090/targets> shows 4 targets UP.
- <http://localhost:3000> → *ML Ops* folder → *ML API — starter* dashboard has live data.

## 2. Generate traffic and watch the dashboard

Write a tiny load generator — either a shell loop or a Python script — that
hits `/predict` with varied inputs a few times a second for a minute.

**Outcome:**

- The Grafana dashboard shows non-zero request rate and p95 latency.
- The *Predictions by class* panel shows multiple classes (iris has 3).

## 3. Add a custom metric

Edit `ml-app/src/api.py`:

1. Add a `Histogram` for prediction latency *only* (i.e. the model call itself,
   not the HTTP overhead). Name it `ml_predict_duration_seconds`.
2. Wrap `model.predict(...)` with `.time()`.
3. Add a panel in Grafana using
   `histogram_quantile(0.95, sum by (le) (rate(ml_predict_duration_seconds_bucket[5m])))`.

**Outcome:** new panel in the dashboard showing model-only latency.

## 4. Write a real alert rule

Edit `prometheus/rules/ml-app.yml`. Pick one of the `TODO(you)` alerts and
implement it. Reload Prometheus without restarting:

```bash
curl -X POST http://localhost:9090/-/reload
```

**Outcome:** <http://localhost:9090/alerts> shows your rule. Induce the
condition (e.g. slow path by `time.sleep`) and watch it fire.

## 5. Bring up GitLab and register the runner

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
./scripts/register-runner.sh <REGISTRATION_TOKEN>
```

**Outcome:** Admin Area → Runners shows one active runner tagged `docker,local`.

## 6. Push the ml-app to GitLab and run the pipeline

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

## 7. Wire the `build` stage to the GitLab Container Registry

1. In the project: Settings → General → Visibility → enable *Container Registry*.
2. Uncomment the `docker login` + `docker push` lines in `.gitlab-ci.yml`.
3. Push. The pipeline will now push the image to the project's registry.

**Outcome:** Packages & Registries → Container Registry shows your image
tagged with the commit SHA.

## 8. Use the Model Registry properly

Right now the API always loads `models:/iris-classifier/None` (= latest
version). Learn the registry workflow:

1. In the MLflow UI, find your model, transition version 1 to **Staging**.
2. Set `ML_MODEL_STAGE=Staging` in `.env` and `docker compose up -d ml-app`.
3. Re-run `python -m src.train` locally to register version 2.
4. Promote version 2 to Staging; demote version 1 to Archived.
5. Restart `ml-app`. `/health` should show the new `model_source`.

**Outcome:** you can flip which model version serves traffic *without*
changing code — only the registry.

## 9. Move training out of the API container

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

## 10. Harden it a bit

Pick at least two:

- Swap MLflow's SQLite for Postgres; swap the artifact FS for MinIO.
- Add Alertmanager to compose; route alerts to Slack or a webhook.
- Build a *Container health* Grafana dashboard using cAdvisor metrics
  (`container_cpu_usage_seconds_total`, `container_memory_usage_bytes`, …).
  Hint: Grafana's dashboard ID 14282 is a solid cAdvisor starter.
- Add input validation (`pydantic` constraints) and a `422`-focused test.
- Add `ruff format` as a CI check; add `mypy` if you're feeling brave.

---

When you've done 1–9 you've covered: CI/CD for ML, experiment tracking, model
registry, service monitoring, alerting, and container health. That's the core
of MLOps. The rest — feature stores, data versioning, batch scoring, shadow
deploys — builds on this base.
