#!/usr/bin/env python3
"""Fork tree — parent keeps children alive (and can leave a zombie).

TODO(you): enable Tree view in htop and map PPID → children — see TASKS.md §4
TODO(you): toggle MAKE_ZOMBIE=1 and find the Z state — see TASKS.md §4
"""

from __future__ import annotations

import os
import signal
import sys
import time


LAB_TAG = os.environ.get("LAB_TAG", "htop-lab")
CHILDREN = int(os.environ.get("FORK_CHILDREN", "3"))
MAKE_ZOMBIE = os.environ.get("MAKE_ZOMBIE", "0") == "1"
os.environ["HTOP_LAB_ROLE"] = "fork"


def _child_work(idx: int) -> None:
    os.environ["HTOP_LAB_ROLE"] = f"fork-child-{idx}"
    # child ignores SIGINT so only parent teardown cleans up via SIGTERM path
    while True:
        time.sleep(30)


def _handle(signum: int, _frame: object) -> None:
    print(f"[{LAB_TAG}:fork] parent got signal {signum}, exiting", flush=True)
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    print(
        f"[{LAB_TAG}:fork] pid={os.getpid()} spawning {CHILDREN} children "
        f"(MAKE_ZOMBIE={MAKE_ZOMBIE})",
        flush=True,
    )
    kids: list[int] = []
    for i in range(CHILDREN):
        pid = os.fork()
        if pid == 0:
            if MAKE_ZOMBIE and i == 0:
                # exit immediately; parent deliberately never wait()s → zombie
                os._exit(0)
            _child_work(i)
            os._exit(0)
        kids.append(pid)
        print(f"[{LAB_TAG}:fork] child[{i}] pid={pid}", flush=True)

    # Keep parent alive; do NOT wait() when MAKE_ZOMBIE so Z state appears
    while True:
        if not MAKE_ZOMBIE:
            # reap any accidental exits so we stay clean in default mode
            try:
                while True:
                    wpid, _ = os.waitpid(-1, os.WNOHANG)
                    if wpid == 0:
                        break
            except ChildProcessError:
                pass
        time.sleep(2)


if __name__ == "__main__":
    main()
