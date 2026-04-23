# Prometheus — TODO

The starter config has 4 scrape jobs and 1 example alert. Everything below is your job.

## Alert rules (`rules/ml-app.yml`)

The file already lists hints in `TODO(you)` comments. Implement at least:

- [ ] **HighLatencyP95** — `ml-app` p95 latency > 0.5s for 5m. Hint: `histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{job="ml-app"}[5m])))`
- [ ] **HighErrorRate** — 5xx ratio > 5% for 5m.
- [ ] **NoTrafficForModel** — `rate(ml_predictions_total[10m]) == 0` for 15m. (Severity `warning`, not critical.)
- [ ] Reload Prometheus without restart: `curl -X POST http://localhost:9090/-/reload`. Verify rules at <http://localhost:9090/rules>.
- [ ] Induce each alert (slow handler, fake 500, no traffic) and watch them go from `inactive` → `pending` → `firing` at <http://localhost:9090/alerts>.

## Recording rules (new file `rules/recording.yml`)

Recording rules pre-compute expensive queries so dashboards stay fast. Build:

- [ ] `job:http_request_rate:5m` = `sum by (job) (rate(http_requests_total[5m]))`
- [ ] `job:http_error_rate:5m` = ratio of 5xx over total per job.
- [ ] `instance:node_cpu_util:5m` = 1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m]))
- [ ] Update one Grafana panel to use the recording rule and confirm the render is faster.

## Service health

- [ ] Add the [blackbox exporter](https://github.com/prometheus/blackbox_exporter) as a new service in `docker-compose.yml`. Probe `http://ml-app:8000/health` every 30s. Alert if `probe_success == 0` for 2m. This catches issues a self-scrape can't (e.g. ml-app's `/metrics` works but `/predict` is broken).

## Alertmanager

- [ ] Add `alertmanager` (image `prom/alertmanager:v0.27`) to `docker-compose.yml` with a config file at `prometheus/alertmanager.yml`.
- [ ] Uncomment the `alerting:` block in `prometheus.yml` and point at it.
- [ ] Configure one receiver (webhook is easiest — `https://webhook.site` for testing). Then a Slack/email receiver if you have one.
- [ ] Write a routing tree: `severity=critical` → page, `severity=warning` → ticket. Use `group_by` to bundle related alerts.
- [ ] Test silences: silence an alert via the Alertmanager UI, confirm it stops firing.

## Scrape config

- [ ] Tune `scrape_interval` per job. `gitlab` (when you add it) is fine at 60s; `ml-app` should stay at 15s.
- [ ] Add `metrics_path` and `scheme` to make jobs explicit (it's an exercise in reading the [scrape_config docs](https://prometheus.io/docs/prometheus/latest/configuration/configuration/#scrape_config)).
- [ ] Add a `relabel_configs` step that drops `go_*` metrics from `ml-app` (they bloat cardinality and you don't care about them locally).
- [ ] Add scraping for the GitLab built-in metrics endpoint (it exposes Prometheus format on `/-/metrics` once enabled in `gitlab.rb`).

## Storage / retention

- [ ] Bump `--storage.tsdb.retention.time` from `7d` to `30d` (it's set in `docker-compose.yml`, not here). Watch the `prometheus_tsdb_*` metrics on Prometheus's own scrape — note disk growth.
- [ ] Add `--storage.tsdb.retention.size=10GB` as a safety cap.

## Hardening (when you're ready)

- [ ] Put Prometheus behind a reverse proxy with basic auth (or just add a `--web.config.file` with `basic_auth_users`).
- [ ] Externalise scrape secrets (e.g. for protected endpoints) via env vars + `${VAR}` substitution. Don't commit them.
