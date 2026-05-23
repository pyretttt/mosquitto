# Kubernetes leg (kind)

This is a deliberately thin slice of "the real thing": one local cluster,
one Helm chart for the FastAPI gateway, raw manifests for the stateful
dependencies. Use it on **Day 3** of the fast track.

## Prerequisites

```
brew install kind kubectl helm
```

## Build and load the API image

```bash
docker build -t little-ml-story-api:latest -f apps/api/Dockerfile .
kind load docker-image little-ml-story-api:latest --name little-ml-story
```

## Bring everything up

```bash
make kind-up
make helm-install
kubectl -n mlops get pods -w
```

## Smoke test

The kind config maps cluster `NodePort 30080` to host `localhost:8000`:

```bash
curl -fsS http://localhost:8000/livez
```

(The Triton deployment ships with an `emptyDir`, so `/predict` will 502
until you populate the model repository — that's the next advanced
exercise: build a model-loading initContainer or push the model to a
ConfigMap / PVC. The chart itself is what we're learning here.)

## Try the autoscaler

```bash
helm upgrade api infra/k8s/helm/api --set autoscaling.enabled=true \
  --reuse-values --namespace mlops
kubectl -n mlops get hpa -w
```

## Tear down

```bash
make helm-uninstall
make kind-down
```
