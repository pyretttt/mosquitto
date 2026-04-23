# Grafana — TODO

The provisioned starter has exactly **one panel** (`ml-app up`). Everything
else is your job. Build dashboards in the UI first, then export the JSON
(`Share → Export → Save to file`) and drop it into `grafana/dashboards/` so
it's reproducible in git.

## Provisioning

- [ ] Re-read `provisioning/datasources/prometheus.yml` and the comments at the top — there are 4 things to do there.
- [ ] Re-read `provisioning/dashboards/dashboards.yml` — 3 more things there.
- [ ] Add an `alerting/` provisioning folder for unified alerting (rules, contact points, notification policies).

## ML API dashboard (extend the starter)

Add these panels to `dashboards/ml-api.json`. You'll first need to wire `/metrics` in `ml-app` (see `ml-app/TODO.md`). Useful queries:

- [ ] **Request rate by handler + status** — `sum by (handler, status) (rate(http_requests_total{job="ml-app"}[1m]))`
- [ ] **p50 / p95 / p99 HTTP latency** — `histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{job="ml-app"}[5m])))`
- [ ] **Error rate (4xx, 5xx)** — `sum(rate(http_requests_total{job="ml-app",status=~"5.."}[5m])) / sum(rate(http_requests_total{job="ml-app"}[5m]))`
- [ ] **Predictions per second by class** — `sum by (predicted_class) (rate(ml_predictions_total[1m]))` (requires the custom counter from `ml-app/TODO.md`)
- [ ] **Model-only latency p95** — `histogram_quantile(0.95, sum by (le) (rate(ml_predict_duration_seconds_bucket[5m])))`

## Container health dashboard (new file)

Create `dashboards/containers.json`. Driven by cAdvisor, scraped on port 8080 inside the network.

- [ ] **CPU per container** — `sum by (name) (rate(container_cpu_usage_seconds_total{name=~".+"}[1m]))`
- [ ] **Memory per container** — `container_memory_usage_bytes{name=~".+"}`
- [ ] **Network RX/TX per container** — `rate(container_network_receive_bytes_total{name=~".+"}[1m])`
- [ ] **Restart count** — `changes(container_start_time_seconds{name=~".+"}[1h])`
- [ ] Add a **template variable** `$container` (Query: `label_values(container_memory_usage_bytes, name)`) so panels are filterable.
- [ ] (Shortcut) Import community dashboard ID **14282** from grafana.com, then strip it down to what you actually use. Reading someone else's dashboard JSON is a learning exercise in itself.

## Host dashboard

- [ ] Build a small node-exporter dashboard (CPU, mem, disk, FS usage). Or import dashboard ID **1860** and trim. Note: on macOS the metrics describe the Linux VM Docker runs in, not your Mac.

## Alerting (Grafana-native, separate from Prometheus rules)

- [ ] Create a contact point (start with a webhook to `https://webhook.site/...` or your own Slack).
- [ ] Create one alert rule that mirrors the Prometheus `MlAppDown` rule, but using Grafana's alerting UI. Compare the two — when would you use one vs the other?
- [ ] Add a notification policy that routes `severity=critical` to your contact point.

## SLO dashboard (stretch)

- [ ] Define an SLO for `ml-app`: e.g. 99% of `/predict` requests under 200 ms, computed over 28 days.
- [ ] Build a panel showing **error budget remaining** (good_events / total_events vs target).
- [ ] Read about the [Sloth](https://github.com/slok/sloth) operator for k8s — it generates Prometheus rules + Grafana dashboards from an SLO YAML. Worth knowing about for when you migrate.
