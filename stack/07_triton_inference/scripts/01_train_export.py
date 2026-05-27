"""Train a tiny sklearn model, export to ONNX, write config.pbtxt.

Result:
  model_repository/
  └── churn_onnx/
      ├── config.pbtxt
      └── 1/
          └── model.onnx
"""

import pathlib
import textwrap

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

REPO = pathlib.Path("model_repository/churn_onnx")
N_FEATURES = 8


def train() -> LogisticRegression:
    X, y = make_classification(
        n_samples=5000, n_features=N_FEATURES, n_informative=5,
        random_state=42,
    )
    model = LogisticRegression(max_iter=500).fit(X, y)
    print(f"train acc = {model.score(X, y):.3f}")
    return model


def export(model: LogisticRegression, out: pathlib.Path) -> None:
    initial_type = [("input", FloatTensorType([None, N_FEATURES]))]
    onx = convert_sklearn(
        model,
        initial_types=initial_type,
        options={LogisticRegression: {"zipmap": False}},  # plain tensor outputs
        target_opset=17,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(onx.SerializeToString())
    print(f"wrote {out} ({out.stat().st_size/1024:.1f} KB)")


def write_config() -> None:
    # See: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html
    cfg = textwrap.dedent(f"""
        name: "churn_onnx"
        backend: "onnxruntime"
        max_batch_size: 0

        input [
          {{
            name: "input"
            data_type: TYPE_FP32
            dims: [ -1, {N_FEATURES} ]
          }}
        ]
        output [
          {{
            name: "label"
            data_type: TYPE_INT64
            dims: [ -1 ]
          }},
          {{
            name: "probabilities"
            data_type: TYPE_FP32
            dims: [ -1, 2 ]
          }}
        ]

        # Serve the two newest versions side by side (used in step 5 of README).
        version_policy: {{ latest: {{ num_versions: 2 }} }}

        instance_group [ {{ count: 1 kind: KIND_CPU }} ]
    """).strip() + "\n"
    (REPO / "config.pbtxt").write_text(cfg)
    print(f"wrote {REPO/'config.pbtxt'}")


def main() -> None:
    model = train()
    export(model, REPO / "2" / "model.onnx")
    write_config()
    # quick sanity check that the ONNX runs
    import onnxruntime as ort
    sess = ort.InferenceSession(str(REPO / "2" / "model.onnx"))
    sample = np.random.randn(2, N_FEATURES).astype(np.float32)
    outs = sess.run(None, {"input": sample})
    print("onnx sanity check outputs:")
    for name, val in zip([o.name for o in sess.get_outputs()], outs):
        print(" ", name, val.tolist())


if __name__ == "__main__":
    main()
