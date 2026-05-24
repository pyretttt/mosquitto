"""Train a *different* model and drop it as version 2 of the same model.

Because Triton was started with --model-control-mode=poll and we're polling
the repo every 5s, no restart is needed: it will detect the new version,
load it, and (per our version_policy) keep both v1 and v2 ready.
"""

import pathlib

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

REPO = pathlib.Path("model_repository/churn_onnx")
N_FEATURES = 8


def main() -> None:
    X, y = make_classification(
        n_samples=5000, n_features=N_FEATURES, n_informative=6, random_state=7,
    )
    # different algorithm -> different predictions = clearly a "new model"
    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=7).fit(X, y)
    print(f"v2 train acc = {model.score(X, y):.3f}")

    onx = convert_sklearn(
        model,
        initial_types=[("input", FloatTensorType([None, N_FEATURES]))],
        # GBC pipelines emit a ZipMap by default; disabling keeps a plain tensor
        # that matches our existing config.pbtxt output schema.
        options={GradientBoostingClassifier: {"zipmap": False}},
        target_opset=17,
    )

    out = REPO / "2" / "model.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(onx.SerializeToString())
    print(f"wrote {out}")
    print("\nNow wait ~5s for Triton to pick it up, then:")
    print("  curl -s localhost:8000/v2/repository/index | jq")
    print("  python scripts/02_call_http.py --version 2")


if __name__ == "__main__":
    main()
