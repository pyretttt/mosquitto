"""Call Triton via the HTTP/JSON inference protocol (KFServing v2)."""

import argparse
import time

import numpy as np
import requests

MODEL = "churn_onnx"
URL = "http://localhost:8000"
N_FEATURES = 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None, help="model version, e.g. 1, 2")
    args = ap.parse_args()

    np.random.seed(0)
    batch = np.random.randn(3, N_FEATURES).astype(np.float32)

    endpoint = f"{URL}/v2/models/{MODEL}"
    if args.version:
        endpoint += f"/versions/{args.version}"
    endpoint += "/infer"

    payload = {
        "inputs": [{
            "name": "input",
            "shape": list(batch.shape),
            "datatype": "FP32",
            "data": batch.flatten().tolist(),
        }],
        "outputs": [
            {"name": "label"},
            {"name": "probabilities"},
        ],
    }

    t0 = time.perf_counter()
    r = requests.post(endpoint, json=payload, timeout=10)
    r.raise_for_status()
    dt = (time.perf_counter() - t0) * 1000

    out = r.json()
    print(f"endpoint = {endpoint}")
    print(f"latency  = {dt:.1f} ms")
    for o in out["outputs"]:
        print(f"  {o['name']:14s} shape={o['shape']} data={o['data']}")


if __name__ == "__main__":
    main()
