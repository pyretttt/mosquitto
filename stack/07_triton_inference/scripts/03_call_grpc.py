"""Call Triton via gRPC using the official tritonclient. Usually 2x faster
than HTTP/JSON because of protobuf framing + persistent connection."""

import time

import numpy as np
import tritonclient.grpc as triton_grpc
from tritonclient.utils import np_to_triton_dtype

MODEL = "churn_onnx"
URL = "localhost:8001"
N_FEATURES = 8


def main() -> None:
    client = triton_grpc.InferenceServerClient(url=URL)
    assert client.is_server_ready()
    assert client.is_model_ready(MODEL)

    np.random.seed(0)
    batch = np.random.randn(5, N_FEATURES).astype(np.float32)

    inp = triton_grpc.InferInput("input", batch.shape, np_to_triton_dtype(batch.dtype))
    inp.set_data_from_numpy(batch)

    outs = [
        triton_grpc.InferRequestedOutput("label"),
        triton_grpc.InferRequestedOutput("probabilities"),
    ]

    t0 = time.perf_counter()
    response = client.infer(model_name=MODEL, inputs=[inp], outputs=outs)
    dt = (time.perf_counter() - t0) * 1000

    label = response.as_numpy("label")
    probs = response.as_numpy("probabilities")

    print(f"gRPC latency = {dt:.1f} ms")
    print(f"label = {label.tolist()}")
    print(f"probs = {probs.tolist()}")


if __name__ == "__main__":
    main()
