# Grafana (dashboards & alerting as code)

Grafana runs **inside** the `kube-prometheus-stack` (see `charts/monitoring`).
We don't configure it by clicking in the UI and hoping — everything is
provisioned from files so it's reproducible in git.

## Where the files actually live

So Helm can package them, the provisioning source files live **inside the
monitoring chart**:

```
charts/monitoring/files/
├── dashboards/
│   └── model-server.json        # dashboard, loaded by the sidecar
└── alerting/
    ├── rules.yaml               # Grafana unified-alerting rule
    ├── contact-points.yaml      # where alerts go (webhook to start)
    └── policies.yaml            # routing tree
```

The chart turns each file into a ConfigMap labelled `grafana_dashboard` /
`grafana_alert`. The Grafana **sidecar** (enabled in
`charts/monitoring/values.yaml` → `grafana.sidecar`) discovers those ConfigMaps
and loads them. No restarts, no manual import.

The Prometheus **datasource** is provisioned inline in the same values file
(`grafana.datasources`), pinned to `uid: prometheus` and `editable: false` so
dashboards never silently break from drift.

## Dashboards-as-code workflow

1. Build/iterate on a dashboard in the Grafana UI.
2. `Share → Export → Save to file` (export for sharing externally → off; we
   reference the provisioned datasource uid `prometheus`).
3. Drop the JSON into `charts/monitoring/files/dashboards/`.
4. `mise run monitoring-local` (helm upgrade) — the sidecar reloads it.

## Two alerting paths — know the difference

| | Prometheus rules + Alertmanager | Grafana unified alerting |
| --- | --- | --- |
| Defined in | `PrometheusRule` CRD (`charts/monitoring/templates/`) | `files/alerting/*.yaml` |
| Evaluated by | Prometheus | Grafana |
| Best for | infra/SRE alerts, GitOps, multi-replica HA | cross-datasource, UI-driven, image-in-alert |

This scaffold ships **both** a `ModelServerDown` example so you can compare
them. See `TASKS.md` §1–3.

## Security / persistence

Handled in `charts/monitoring/values.yaml` under `grafana:` — admin secret,
`grafana.ini` hardening, PVC persistence. See `TASKS.md` §2 and
`docs/ADMIN_TASKS.md`.
