"""
demo-app — tiny FastAPI backend for the Kubernetes secrets lab.

Reads non-secret config from env (ConfigMap) and credentials from env (Secret).
Does not talk to Vault — that lab is ops/vault_proj.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

app = FastAPI(title="secrets-lab-demo", version="0.1.0")


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "****"
    return value[:2] + "…" + value[-2:]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def config() -> dict[str, str | None]:
    """Public-ish config — typically sourced from a ConfigMap."""
    return {
        "app_mode": os.getenv("APP_MODE"),
        "log_level": os.getenv("LOG_LEVEL"),
        "welcome": os.getenv("WELCOME_MESSAGE"),
    }

def get_secrets():
    res = dict()
    with open("/etc/secret-volume/API_TOKEN") as token:
        res["token"] = token.read()
    with open("/etc/secret-volume/DB_PASSWORD") as db_password:
        res["db_password"] = db_password.read()
    return res

@app.get("/secret-status")
def secret_status() -> dict[str, object]:
    """
    Prove a Secret was mounted without leaking the raw value.

    TODO(you): after switching Secret sources (plain / SOPS / sealed / ESO),
    restart the pod and confirm source_hint + token_present change — TASKS.md
    """
    token = os.getenv("API_TOKEN", "")
    db_password = os.getenv("DB_PASSWORD", "")
    source_hint = os.getenv("SECRET_SOURCE", "unset")

    if not token:
        raise HTTPException(
            status_code=503,
            detail="API_TOKEN missing — wire a Secret (TASKS.md §1+)",
        )

    return {
        "source_hint": source_hint,
        "token_present": bool(token),
        "token_masked": token,
        "db_password_present": bool(db_password),
    }
