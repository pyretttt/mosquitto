# htop — Master Task List

Start here. Tasks are grouped by concern and ordered roughly easiest → hardest
within each group. Inline `TODO(you)` markers in the code point back to these
items. Tick them off as you go.

**Time box:** ~1–2 days × 2–3 hours. Suggested order: §0 → §1 → §2 → §3 → §4 →
§5 → §6 optional.

Legend: `[ ]` todo · `[~]` partially scaffolded · `[x]` done

Pair each section with `docs/CONCEPTS.md`.

---

## Problem

You share a workstation (or a beefy “training” laptop) with batch jobs, data
prep scripts, and long-lived workers. Something is wrong: the fan spins, the UI
stutters, a pipeline “runs” for hours with no progress, or `df` looks fine but
everything feels stuck on disk. Cloud dashboards are not in front of you —
only the host.

**Done looks like:** you can open htop, explain the header meters and key
columns out loud, find the lab workloads by filter/tree, match each symptom
class (CPU / mem / IO / hang / zombie) to the right cues, and intervene with
renice / TERM / `lsof` without shotgun-killing the wrong PID. You leave a short
runbook note in `docs/notes.md`.

---

## Learning outcomes

- Drive htop (setup, filter, sort, tree, kill, renice)
- Interpret CPU%, RES/VIRT/SWAP, process state, nice, and parent/child trees
- Troubleshoot CPU-bound, memory-pressure, IO-heavy, hung, and zombie cases
- Use `ps` / `pgrep` / `kill` / `renice` / `lsof` alongside htop
- Connect host process views to MLOps/DevOps failure modes

---

## 0. Bootstrap

- [ ] Install mise if needed; from `ops/htop` run `mise install`.
- [ ] Install htop (`brew install htop` or `sudo apt install htop`).
- [ ] `mise run bootstrap` and `mise run verify-tools`.
- [ ] **Verify:** `htop --version` and `python --version` print without errors.
- [ ] Skim `README.md` and `docs/CONCEPTS.md` §1–2.

---

## 1. htop orientation

Get comfortable before the machine gets noisy.

- [ ] **Task:** run `mise run htop` (or plain `htop`). Open setup (`F2`): note
      which CPU/memory meters you have. Enable or confirm columns:
      `PID`, `USER`, `PRI`, `NI`, `S`, `CPU%`, `MEM%`, `TIME+`, `Command`.
- [ ] **Task:** practice filter (`F4`) and search (`F3`). Sort by CPU, then by MEM.
- [ ] **Task:** toggle tree view (`F5` / `t`) and hide userland threads (`H`) —
      notice how Python thread noise changes.
- [ ] **Task:** record your preferred setup in `docs/notes.md`
      (`TODO(you)` there and in `scripts/launch-htop.sh`).
- [ ] **Verify:** you can find your shell’s PID in htop within ~10 seconds.

---

## 2. CPU-bound noise

Training loops and busy scrapers look like this.

- [ ] **Task:** `mise run cpu-start` then `mise run htop` (filter `cpu_burn`).
- [ ] **Task:** watch per-core meters and the process `CPU%`. How many cores does
      one single-threaded burner consume?
- [ ] **Task:** from another terminal, confirm with:
      ```bash
      pgrep -af cpu_burn
      ps -o pid,ni,state,pcpu,pmem,command -p "$(pgrep -f cpu_burn.py | head -1)"
      ```
- [ ] **Task:** in htop, renice the burner to `+10` (`r`). Does `CPU%` drop when
      something else competes? (Optional: start a second `cpu-start` after
      editing the script to allow two — or run `python workloads/cpu_burn.py &`
      manually.)
- [ ] **Verify:** you can explain user vs system vs nice time at a high level
      (see `docs/CONCEPTS.md` §2–3).
- [ ] Stop when done: `mise run workloads-stop` (or leave running for §3).

---

## 3. Memory pressure & IO wait

- [ ] **Task:** `mise run mem-start`. In htop, watch `RES` climb in steps. Compare
      `VIRT` vs `RES` — why can VIRT be large while RES tracks the hog?
- [ ] **Task:** optionally raise pressure: `MEM_TARGET_MB=512 mise run mem-start`
      (stop first). Stay inside your machine’s free RAM — do **not** aim to OOM
      your desktop session.
- [ ] **Task:** `mise run io-start`. Look for IO-wait in the header (if shown)
      and state `D` flashes. Tail `.run/io.log` while watching htop.
- [ ] **Task:** list open files for the IO process:
      ```bash
      lsof -p "$(pgrep -f io_churn.py | head -1)" | head
      ```
- [ ] **Verify:** write 2–3 lines in `docs/notes.md` distinguishing “CPU pegged”
      vs “RAM climbing” vs “disk busy / IO wait”.
- [ ] `mise run workloads-stop`.

---

## 4. Trees, parents, and zombies

MLOps jobs often spawn loaders / workers; orphaned children and zombies hide in
flat lists.

- [ ] **Task:** `mise run fork-start`. Enable **tree** view. Find the parent
      `fork_tree.py` and its children. Note PPIDs with:
      ```bash
      ps -o pid,ppid,state,command -p "$(pgrep -f fork_tree.py | tr '\n' ',' | sed 's/,$//')"
      ```
      (or inspect each PID from htop).
- [ ] **Task:** stop workloads, then start a zombie case:
      ```bash
      MAKE_ZOMBIE=1 mise run fork-start
      ```
      Find state `Z` in htop. Who is the parent? Why is the child still listed?
- [ ] **Task:** `mise run workloads-stop`. Confirm zombies/children are gone
      (`mise run workloads-status` / `pgrep`).
- [ ] **Verify:** you can say out loud what a zombie is (exited child, not yet
      `wait()`ed) and that killing the zombie itself does nothing useful — fix
      or kill the parent.

---

## 5. Hung worker, signals, and safe intervention

- [ ] **Task:** `mise run hung-start`. Filter to `hung_worker`. Note state `S`
      and near-zero CPU.
- [ ] **Task:** confirm the held lock file:
      ```bash
      lsof -p "$(pgrep -f hung_worker.py | head -1)"
      ls -la .data/
      ```
- [ ] **Task:** from htop send `SIGTERM` (`k`, choose 15). Confirm the process
      exits and `.data/hung-*.lock` behavior (file may remain — optional cleanup).
- [ ] **Task:** edit `scripts/workloads.sh` so `start_one` launches via `nice -n 10`
      (`TODO(you)` in that file). Restart cpu burner and confirm `NI` is `10` in
      htop.
- [ ] **Verify:** `mise run workloads-stop` leaves `workloads-status` clean.

---

## 6. MLOps / DevOps transfer (optional hardening)

No cluster required — mental model only, plus a tiny write-up.

- [ ] **Task:** pick one real scenario you have seen (stuck training job, noisy
      CI runner, “pod is Running but app hung”). Map which htop cues you would
      seek **on the node**, and what the K8s-level analogue would be
      (`kubectl top`, container PID namespaces, throttling).
- [ ] **Task:** add a half-page “host triage” runbook to `docs/notes.md`:
      filter → sort → tree → `lsof`/`ps` → TERM → escalate.
- [ ] **Optional:** compare `htop` vs `top` vs (if installed) `glances` for the
      same `cpu_burn` — what is faster for interactive triage?
- [ ] **Verify:** a teammate could follow your runbook without reading TASKS.md.

---

## Stretch ideas (out of scope unless you want them)

- [ ] Observe the same workloads inside a Docker container vs on the host PIDs
- [ ] Add a cgroup v2 CPU max to the cpu burner and watch throttling
- [ ] Capture `ps`/`htop` batch output in CI as an artifact for a fake “incident”
