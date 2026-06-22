"""Thin S3 helper around boto3, pointed at MinIO (or real S3) via env.

The only thing the rest of the app needs is "download the model artifact to a
local path". Kept deliberately small so swapping MinIO -> real S3 is just an
endpoint/credentials change (no code change).
"""

from __future__ import annotations

import logging

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from .config import settings

log = logging.getLogger(__name__)


def _client():
    # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are picked up from env by boto3.
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        region_name=settings.aws_region,
        config=Config(signature_version="s3v4"),
    )


def download_model(local_path: str) -> bool:
    """Download the model artifact to `local_path`.

    Returns True on success, False if the object doesn't exist (so the caller
    can fall back gracefully instead of crashing the pod).
    """
    bucket = settings.s3_models_bucket
    # MODEL_KEY may include the bucket prefix ("models/model.joblib"); strip it.
    key = settings.model_key
    if key.startswith(f"{bucket}/"):
        key = key[len(bucket) + 1 :]

    try:
        _client().download_file(bucket, key, local_path)
        log.info("downloaded model s3://%s/%s -> %s", bucket, key, local_path)
        return True
    except ClientError as e:
        # 404 / NoSuchKey -> no model trained yet. Anything else: log + fall back.
        log.warning("could not download model s3://%s/%s: %s", bucket, key, e)
        return False
