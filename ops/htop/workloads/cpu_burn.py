#!/usr/bin/env python3
"""CPU burner — pegs one core with a tight loop.

TODO(you): tune WORKERS / affinity notes in TASKS.md §2 — see TASKS.md §2
"""

from __future__ import annotations

import os
import signal
import sys
import time


LAB_TAG = os.environ.get("LAB_TAG", "htop-lab")
# Set process title-ish via env so pgrep/htop filters can find us
os.environ["HTOP_LAB_ROLE"] = "cpu"


def _handle(_signum: int, _frame: object) -> None:
    print(f"[{LAB_TAG}:cpu] shutting down", flush=True)
    sys.exit(0)


def burn() -> None:
    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    print(f"[{LAB_TAG}:cpu] pid={os.getpid()} burning CPU", flush=True)
    x = 0
    while True:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        # tiny yield so the process stays killable on busy systems
        if x % 50_000_000 == 0:
            time.sleep(0)


if __name__ == "__main__":
    burn()
