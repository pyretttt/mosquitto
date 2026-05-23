import io

import numpy as np
import pytest
from PIL import Image

from apps.api.app.triton_client import IMAGE_SIZE, preprocess_image, softmax


def _make_image(size: tuple[int, int] = (320, 200), color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_preprocess_shapes_and_dtype():
    arr = preprocess_image(_make_image())
    assert arr.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert arr.dtype == np.float32


def test_preprocess_handles_grayscale():
    img = Image.new("L", (256, 256), 128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    arr = preprocess_image(buf.getvalue())
    assert arr.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_softmax_sums_to_one():
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    assert probs.shape == (3,)
    assert pytest.approx(probs.sum(), abs=1e-6) == 1.0


def test_preprocess_empty_payload_fails():
    with pytest.raises(Exception):
        preprocess_image(b"")
