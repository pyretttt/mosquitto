"""
demo-app — tiny FastAPI backend for the Vault lab.

Scaffold behavior:
  - /health always works
  - /secret tries to read from Vault; falls back to a clearly-fake value until
    you finish the TODO(you) auth + read path (TASKS.md §3)
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI(title="vault-lab-demo", version="0.1.0")

VAULT_ADDR = os.getenv("VAULT_ADDR", "http://vault.vault.svc:8200").rstrip("/")
VAULT_SECRET_PATH = os.getenv("VAULT_SECRET_PATH", "secret/data/demo/db")
# Static token is OK for first bring-up; replace with K8s auth (TODO below).
VAULT_TOKEN = os.getenv("VAULT_TOKEN", "")


def _read_k8s_sa_jwt() -> str | None:
    path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def vault_token_via_kubernetes() -> str:
    """
    Authenticate to Vault with this pod's ServiceAccount JWT.

    TODO(you): finish K8s auth — see TASKS.md §3 and scripts/k8s-bootstrap.sh
      1. Ensure auth/kubernetes is configured and role `demo-app` exists.
      2. POST to {VAULT_ADDR}/v1/auth/kubernetes/login
         JSON: {"role": "demo-app", "jwt": "<sa token>"}
      3. Return client_token from the response.
    """
    jwt = _read_k8s_sa_jwt()
    if not jwt:
        raise RuntimeError("no ServiceAccount JWT mounted")

    role = os.getenv("VAULT_K8S_ROLE", "demo-app")
    # TODO(you): replace this stub with a real httpx post to Vault login
    raise NotImplementedError(
        f"wire kubernetes login for role={role} against {VAULT_ADDR} — TASKS.md §3"
    )


def get_vault_client_token() -> str:
    if VAULT_TOKEN:
        return VAULT_TOKEN
    return vault_token_via_kubernetes()


def read_kv_secret(token: str) -> dict[str, Any]:
    """
    Read KV v2 secret. Path should look like secret/data/demo/db
    (engine mount + /data/ + secret path).
    """
    url = f"{VAULT_ADDR}/v1/{VAULT_SECRET_PATH.lstrip('/')}"
    headers = {"X-Vault-Token": token}
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(url, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"vault read failed: {resp.status_code} {resp.text[:200]}",
        )
    payload = resp.json()
    return payload.get("data", {}).get("data", {})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/secret")
def secret() -> dict[str, Any]:
    """
    Returns non-sensitive fields to prove Vault access works.
    Never log the raw password.
    """
    try:
        token = get_vault_client_token()
        data = read_kv_secret(token)
    except NotImplementedError as exc:
        # Scaffold fallback so the chart can start before you finish auth.
        return {
            "source": "scaffold-fallback",
            "username": "not-from-vault-yet",
            "password_present": False,
            "hint": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 — lab app surfaces errors to the learner
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # TODO(you): after rotation lab (TASKS.md §4), show secret "version" from
    # Vault metadata so you can see cut-over between KV versions.
    return {
        "source": "vault",
        "username": data.get("username"),
        "password_present": bool(data.get("password")),
    }


# TODO(you): optional HTTPS — load cert/key issued by Vault PKI (TASKS.md §3d)
# and run uvicorn with ssl_certfile/ssl_keyfile, or terminate TLS at Ingress.
