#!/usr/bin/env python3
"""Hung worker — sleeps forever (or blocks on a lock file).

Looks "stuck" in htop (S / sleeping) while holding a file — practice with lsof.

TODO(you): find this PID via htop filter + confirm open files with lsof — see TASKS.md §5
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path


LAB_TAG = os.environ.get("LAB_TAG", "htop-lab")
DATA_DIR = Path(os.environ.get("DATA_DIR", ".data"))
os.environ["HTOP_LAB_ROLE"] = "hung"


def _handle(_signum: int, _frame: object) -> None:
    print(f"[{LAB_TAG}:hung] shutting down", flush=True)
    sys.exit(0)


def hang() -> None:
    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock = DATA_DIR / f"hung-{os.getpid()}.lock"
    f = open(lock, "w")
    f.write(f"held-by={os.getpid()}\n")
    f.flush()
    # keep FD open on purpose
    print(
        f"[{LAB_TAG}:hung] pid={os.getpid()} sleeping with lock open: {lock}",
        flush=True,
    )
    while True:
        time.sleep(60)


if __name__ == "__main__":
    hang()
