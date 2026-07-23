#!/usr/bin/env python3
"""Memory hog — grows RSS in steps until TARGET_MB or killed.

TODO(you): observe SWAP / RES / VIRT in htop while this climbs — see TASKS.md §3
"""

from __future__ import annotations

import os
import signal
import sys
import time


LAB_TAG = os.environ.get("LAB_TAG", "htop-lab")
TARGET_MB = int(os.environ.get("MEM_TARGET_MB", "256"))
STEP_MB = int(os.environ.get("MEM_STEP_MB", "32"))
HOLD_SEC = float(os.environ.get("MEM_HOLD_SEC", "120"))
os.environ["HTOP_LAB_ROLE"] = "mem"

_chunks: list[bytearray] = []


def _handle(_signum: int, _frame: object) -> None:
    print(f"[{LAB_TAG}:mem] shutting down (held {len(_chunks)} chunks)", flush=True)
    sys.exit(0)


def hog() -> None:
    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    print(
        f"[{LAB_TAG}:mem] pid={os.getpid()} targeting ~{TARGET_MB} MiB "
        f"in {STEP_MB} MiB steps",
        flush=True,
    )
    allocated = 0
    while allocated < TARGET_MB:
        chunk = bytearray(STEP_MB * 1024 * 1024)
        # touch pages so RSS actually grows (not just VIRT)
        for i in range(0, len(chunk), 4096):
            chunk[i] = 1
        _chunks.append(chunk)
        allocated += STEP_MB
        print(f"[{LAB_TAG}:mem] RSS~{allocated} MiB", flush=True)
        time.sleep(1)
    print(f"[{LAB_TAG}:mem] holding for {HOLD_SEC}s — watch htop", flush=True)
    time.sleep(HOLD_SEC)
    print(f"[{LAB_TAG}:mem] done", flush=True)


if __name__ == "__main__":
    hog()
