"""End-to-end happy path against a running compose stack.

Expects `make up-all && make migrate` to have completed. Skips if the
service isn't reachable so CI doesn't fail without infra.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

API_URL = os.getenv("API_URL", "http://localhost:8000")
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "cat.jpg"


def _service_up() -> bool:
    try:
        r = httpx.get(f"{API_URL}/livez", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_stack():
    if not _service_up():
        pytest.skip("API not reachable; run `make up-all` first")
    if not FIXTURE.exists():
        pytest.skip(f"missing fixture at {FIXTURE}")


@pytest.mark.asyncio
async def test_predict_then_cache_hit():
    async with httpx.AsyncClient(base_url=API_URL, timeout=30.0) as client:
        with FIXTURE.open("rb") as f:
            r1 = await client.post("/predict", files={"file": ("cat.jpg", f, "image/jpeg")})
        assert r1.status_code == 200, r1.text
        body1 = r1.json()
        assert body1["top_class"]["label"]
        assert body1["cache_hit"] is False

        with FIXTURE.open("rb") as f:
            r2 = await client.post("/predict", files={"file": ("cat.jpg", f, "image/jpeg")})
        assert r2.status_code == 200
        body2 = r2.json()
        assert body2["cache_hit"] is True
        assert body2["top_class"]["class_id"] == body1["top_class"]["class_id"]


@pytest.mark.asyncio
async def test_healthz_reports_all_components_ready():
    async with httpx.AsyncClient(base_url=API_URL, timeout=5.0) as client:
        r = await client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["api"] is True
    assert body["triton"] is True
    assert body["redis"] is True
