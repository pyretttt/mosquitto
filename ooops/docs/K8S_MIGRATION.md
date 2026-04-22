# K8s migration notes

Read this *before* you start translating. The goal is a clean migration, not a
line-by-line port of the compose file.

## What's already k8s-friendly in this scaffold

- Service-to-service DNS (`http://mlflow:5000`, `http://prometheus:9090`): in
  k8s these become the same hostnames via `Service` objects — no code change.
- All config via env vars (no file-based config in `ml-app`): maps to
  `ConfigMap` / `Secret` cleanly.
- MLflow serves artifacts through `--serve-artifacts`, so only one pod touches
  the artifact volume (good for a ReadWriteOnce PVC).
- Prometheus, Grafana, node-exporter, cAdvisor all have mature official
  Helm charts (`kube-prometheus-stack` gives you all four, plus Alertmanager).

## What you'll have to redo

| Compose thing | K8s replacement |
| --- | --- |
| `prom/prometheus` + `grafana/grafana` + `node-exporter` + `cadvisor` | Install the `kube-prometheus-stack` Helm chart — gives you all of it + Alertmanager + sane defaults |
| Hand-written `prometheus.yml` | `ServiceMonitor` / `PodMonitor` CRDs that select pods by label |
| Hand-written Grafana provisioning | `ConfigMap` with the `grafana_dashboard: "1"` label; the chart auto-imports them |
| Named volumes (`mlflow-db`, `mlflow-artifacts`) | `PersistentVolumeClaim`s. Use the cluster's default StorageClass locally (k3d / kind / Docker Desktop all ship one) |
| `gitlab/gitlab-ce` image | Use the official GitLab Helm chart, but **honestly**: for local-only learning the Docker version is fine; the Helm chart is heavy. Keep GitLab in compose, move only the ML stack to k8s first |
| `gitlab-runner` with docker executor + host socket | GitLab Runner Helm chart with `kubernetes` executor (each job runs as a Pod). Much cleaner than DinD |
| `docker build` in the CI job | [Kaniko](https://github.com/GoogleContainerTools/kaniko) or `buildkit` in rootless mode |

## Recommended migration order

1. **Pick a local cluster**: `k3d` is the lightest; Docker Desktop's built-in
   k8s is the least setup. Kind works too.
2. **Migrate MLflow and ml-app first.** Smallest blast radius, no stateful
   dependencies besides a PVC. Write `Deployment` + `Service` + `ConfigMap`
   manifests by hand — it's instructive.
3. **Add ingress.** Use the `traefik` or `nginx` ingress controller. Expose
   `ml-app` at `http://ml-app.localtest.me` or similar.
4. **Install `kube-prometheus-stack` via Helm.** Add a `ServiceMonitor` for
   `ml-app` (the chart will auto-discover it via labels). Your existing
   `/metrics` endpoint works unchanged.
5. **Import your Grafana dashboard** as a `ConfigMap` with the sidecar label.
6. **Move the GitLab runner to the `kubernetes` executor.** Keep GitLab itself
   in Docker unless you have a real reason to k8s-ify it locally.
7. **Rewrite `.gitlab-ci.yml`** to use Kaniko for image builds (since you no
   longer have a Docker socket on the runner).

## Likely foot-guns

- **cAdvisor and node-exporter are already inside `kube-prometheus-stack`.**
  Don't install them twice — you'll get duplicate metrics with different labels.
- **Prometheus scrape configs are different.** In k8s you typically don't
  write `static_configs` — you use `ServiceMonitor` + label selectors.
  Plan to rewrite the scrape config, not port it.
- **`host.docker.internal` does not exist in k8s.** If anything in your
  compose relied on it (it shouldn't here), track it down now.
- **`/var/run/docker.sock` mounts don't work.** That's why you need Kaniko.
- **GitLab's external URL.** When GitLab runs in Docker but CI runs in k8s,
  the runner's `url` becomes a cross-boundary call. Use `host.docker.internal`
  from the k8s side (kind/k3d both support it) or put GitLab behind the same
  ingress.
- **Resource requests/limits.** Compose ignores them; k8s won't schedule pods
  without sane requests. Start with something like:

  ```yaml
  resources:
    requests: { cpu: "100m", memory: "256Mi" }
    limits:   { cpu: "500m", memory: "512Mi" }
  ```

  MLflow + Postgres will want more.

## Exit criteria

You've completed the migration when:

- `kubectl get pods -A` is all `Running`.
- `curl http://<ingress>/predict ...` works the same as today.
- A pipeline push to GitLab triggers a k8s-executor job, builds the image with
  Kaniko, pushes to the registry, and (bonus) `kubectl rollout restart` the
  deployment to pick up the new image.
- Grafana dashboards are restored by re-applying the `ConfigMap`, not by
  clicking in the UI.
