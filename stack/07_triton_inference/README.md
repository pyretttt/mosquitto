# 07 — NVIDIA Triton Inference Server (≈ 60 min)

You will export a scikit-learn classifier to **ONNX**, lay out a Triton **model repository**, start Triton in CPU mode, and call it with both `tritonclient` (gRPC) and plain `curl`. Then you'll add a **second model version** and switch traffic.

> Triton on Apple Silicon: the official `nvcr.io/nvidia/tritonserver` images are linux/amd64. The compose file forces `platform: linux/amd64`. It will run under Rosetta — slower but fine for learning. On a real x86 GPU machine it would be 10–100× faster with TensorRT.

## Concepts you must be able to explain

1. **Triton in one sentence.** A production-grade inference server that loads many models (TensorRT, ONNX, PyTorch, TensorFlow, Python, …), serves them over HTTP/gRPC, and does dynamic batching, model versioning, multi-GPU, and metrics — out of the box.
2. **Model repository layout.** Triton is config-driven: it scans a folder, loads everything it finds.
   ```
   model_repository/
   └── churn_onnx/
       ├── config.pbtxt
       └── 1/
           └── model.onnx
       └── 2/                 # version 2
           └── model.onnx
   ```
   The numeric subfolders are versions. `config.pbtxt` defines IO shapes and runtime options.
3. **Backends.** `onnxruntime`, `tensorrt`, `pytorch` (TorchScript), `tensorflow`, `python` (BYO Python "model.py"), `dali`, `openvino`. The backend reads files from the version folder and serves them.
4. **Dynamic batching.** If 50 clients each send a single inference at ~the same time, Triton merges them into one big batch on the GPU. Big win in throughput, tiny cost in latency. Configured per-model in `config.pbtxt`.
5. **Versioning + policies.** `version_policy: latest { num_versions: 1 }`, or `all`, or specific. You deploy v2 next to v1, then flip the policy → zero-downtime model upgrade. Routers like a feature flag can A/B by addressing `model_name/2`.
6. **HTTP vs gRPC.** HTTP/JSON is human-friendly and easy to curl. gRPC + protobuf is ~2× faster and the official `tritonclient` uses it.
7. **Metrics.** Triton exposes Prometheus metrics on port 8002: `nv_inference_request_success`, `nv_inference_compute_infer_duration_us`, `nv_inference_queue_duration_us` — you should know these names.
8. **Why ONNX?** Framework-agnostic exchange format. Train in sklearn / PyTorch / TF → export to ONNX → serve in Triton (or ONNX Runtime alone). Decouples training framework from serving stack.

## Task

### Step 1 — Build the model artifact

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/01_train_export.py
```

This:
- trains a tiny `LogisticRegression` on a synthetic binary task,
- converts it to ONNX with `skl2onnx`,
- writes `model_repository/churn_onnx/1/model.onnx` and the matching `config.pbtxt`.

Inspect the file:

```bash
cat model_repository/churn_onnx/config.pbtxt
```

Read every line — that's exactly the kind of file you'll write in your job.

### Step 2 — Start Triton

```bash
docker compose up -d
docker compose logs -f triton    # wait for "Started GRPCInferenceService at 0.0.0.0:8001"
```

Endpoints (host):
- `http://localhost:8000` — HTTP inference
- `localhost:8001`        — gRPC inference
- `http://localhost:8002/metrics` — Prometheus

### Step 3 — Inspect

```bash
curl -s localhost:8000/v2/health/ready    # -> "" (200)
curl -s localhost:8000/v2/models/churn_onnx | jq
curl -s localhost:8000/v2/models/churn_onnx/config | jq
```

### Step 4 — Call the model

```bash
python scripts/02_call_http.py    # plain HTTP+JSON
python scripts/03_call_grpc.py    # official tritonclient over gRPC
```

You should get the same predictions from both, plus latency.

### Step 5 — Deploy v2 (zero-downtime)

```bash
python scripts/04_add_v2.py
```

It trains a slightly different model and writes `model_repository/churn_onnx/2/model.onnx`. Then run:

```bash
curl -X POST localhost:8000/v2/repository/index | jq
curl -X POST localhost:8000/v2/repository/models/churn_onnx/load   # only needed if explicit model control mode
```

In our config, `version_policy: { latest { num_versions: 2 }}` and Triton's `--model-control-mode=poll` is on — it picks v2 up automatically. Verify:

```bash
curl -s localhost:8000/v2/models/churn_onnx/versions/2 | jq
```

Now call v2 specifically:

```bash
python scripts/02_call_http.py --version 2
```

### Step 6 — Watch the metrics

```bash
curl -s localhost:8002/metrics | grep nv_inference_request
```

Run the call script in a loop — counters increment per model + per version. In production you'd scrape this with Prometheus and chart latency in Grafana.

## Interview questions to rehearse

- "Why Triton over Flask + pickle?" → batching, versioning, multi-framework, GPU scheduling, metrics, gRPC, no Python GIL in hot path. You don't reinvent any of it.
- "Walk me through deploying a new model version." → drop new version folder in repo → poll mode picks it up → flip routing / version_policy → monitor metrics → keep N-1 as instant rollback.
- "What is dynamic batching and when does it help?" → many concurrent small requests on a model whose per-call kernel launch overhead dominates. Doesn't help for batch-size-1 workloads on CPU with tiny models.
- "Triton vs TorchServe vs TFServing vs BentoML?" → Triton: multi-framework, NVIDIA-optimised, mature. TorchServe: PyTorch only. TFServing: TF only. BentoML: Python-first, easier dev UX, less performant for raw inference.
- "What's in config.pbtxt?" → `name`, `platform/backend`, `input`/`output` tensors (name, dtype, dims), `max_batch_size`, optional `dynamic_batching`, `instance_group`, `version_policy`.
- "Why ONNX in the middle?" → portability + the runtime can apply graph-level optimisations independent of the training framework.

## References

- Triton user guide: <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/>
- Model repository docs: <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html>
- `config.pbtxt` reference: <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html>
- `tritonclient` (Python): <https://github.com/triton-inference-server/client>
- ONNX intro: <https://onnx.ai/get-started.html>
- `skl2onnx`: <https://onnx.ai/sklearn-onnx/>
