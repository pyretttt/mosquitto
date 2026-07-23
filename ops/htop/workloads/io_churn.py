#!/usr/bin/env python3
"""IO churn — repeated write/read/delete in DATA_DIR.

TODO(you): correlate Disk R/W or IO-wait with this process — see TASKS.md §3
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path


LAB_TAG = os.environ.get("LAB_TAG", "htop-lab")
DATA_DIR = Path(os.environ.get("DATA_DIR", ".data"))
FILE_MB = int(os.environ.get("IO_FILE_MB", "8"))
ROUNDS = int(os.environ.get("IO_ROUNDS", "0"))  # 0 = forever
os.environ["HTOP_LAB_ROLE"] = "io"

_stop = False


def _handle(_signum: int, _frame: object) -> None:
    global _stop
    _stop = True
    print(f"[{LAB_TAG}:io] shutting down", flush=True)


def churn() -> None:
    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"io-churn-{os.getpid()}.bin"
    blob = os.urandom(min(FILE_MB, 1) * 1024 * 1024)
    # grow blob without another huge urandom if FILE_MB > 1
    if FILE_MB > 1:
        blob = blob * FILE_MB
    print(
        f"[{LAB_TAG}:io] pid={os.getpid()} writing {FILE_MB} MiB cycles under {DATA_DIR}",
        flush=True,
    )
    n = 0
    while not _stop and (ROUNDS == 0 or n < ROUNDS):
        with open(path, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        with open(path, "rb") as f:
            _ = f.read()
        n += 1
        if n % 5 == 0:
            print(f"[{LAB_TAG}:io] round={n}", flush=True)
        time.sleep(0.05)
    path.unlink(missing_ok=True)
    sys.exit(0)


if __name__ == "__main__":
    churn()
