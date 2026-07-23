#!/usr/bin/env bash
# Launch htop with a sensible starting view for this lab.
# Filter syntax differs by platform/build — fall back to plain htop.
#
# TODO(you): add your preferred setup (meters, tree, hide userland threads)
#            and document it in docs/notes.md — see TASKS.md §1
set -euo pipefail

if ! command -v htop >/dev/null 2>&1; then
  echo "htop missing — run: mise run bootstrap" >&2
  exit 1
fi

# Best-effort filter on process command line containing workload names.
# On some builds -F is "filter"; older macOS htop may ignore unknown flags.
if htop --help 2>&1 | grep -qE -- '-F|\-\-filter'; then
  exec htop -F 'cpu_burn|mem_hog|io_churn|fork_tree|hung_worker'
fi

echo "tip: press F4 / type to filter for: cpu_burn  (or mem_hog, io_churn, …)"
exec htop
