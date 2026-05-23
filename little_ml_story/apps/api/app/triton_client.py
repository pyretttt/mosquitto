"""Thin async wrapper around Triton gRPC client.

We use the sync gRPC client offloaded to a thread because the official
async client is still experimental for some versions; this keeps things
robust without pulling in extra deps.
"""

import asyncio
import io
from dataclasses import dataclass

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 224


@dataclass(slots=True)
class TritonConfig:
    url: str
    model_name: str
    model_version: str


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize, center-crop, normalise, transpose to NCHW float32."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img = img.resize((256, 256), Image.BILINEAR)
        left = (256 - IMAGE_SIZE) // 2
        top = (256 - IMAGE_SIZE) // 2
        img = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class TritonInferenceClient:
    """Synchronous Triton client called from an executor."""

    INPUT_NAME = "data"
    OUTPUT_NAME = "resnetv17_dense0_fwd"  # default for the ONNX zoo ResNet50 v1.7 model

    def __init__(self, config: TritonConfig) -> None:
        self._config = config
        self._client = grpcclient.InferenceServerClient(url=config.url)

    def _infer_sync(self, batch: np.ndarray) -> np.ndarray:
        inputs = [grpcclient.InferInput(self.INPUT_NAME, list(batch.shape), "FP32")]
        inputs[0].set_data_from_numpy(batch)
        outputs = [grpcclient.InferRequestedOutput(self.OUTPUT_NAME)]
        response = self._client.infer(
            model_name=self._config.model_name,
            model_version=self._config.model_version,
            inputs=inputs,
            outputs=outputs,
        )
        return response.as_numpy(self.OUTPUT_NAME)

    async def infer(self, batch: np.ndarray) -> np.ndarray:
        return await asyncio.get_running_loop().run_in_executor(None, self._infer_sync, batch)

    def ready(self) -> bool:
        try:
            return bool(
                self._client.is_server_ready()
                and self._client.is_model_ready(self._config.model_name)
            )
        except Exception:
            return False
