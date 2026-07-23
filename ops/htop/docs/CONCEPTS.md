# Concepts — htop & process management

Short theory companion to `TASKS.md`. Aimed at DevOps / MLOps practitioners
who live on shared training boxes, CI runners, and laptop-as-cluster hosts.

## 1. Why process tools matter in MLOps / DevOps

| Situation | What you look for |
| --------- | ----------------- |
| Training job pegs a node | Which PID owns the cores? Parent trainer vs data loader? |
| OOM kills / swap thrash | RES climbing, who else shares the box? |
| “Pipeline is slow” | CPU wait vs IO wait vs runnable queue |
| Stuck deploy / hung worker | State `D`/`S`, open files, PPID tree |
| Noisy neighbor on a shared GPU host | nice/priority, cgroup limits (next step past this lab) |

`htop` is an interactive lens over the same kernel accounting that `ps`,
`/proc`, and container runtimes expose. Learn the lens here; the metrics
transfer to `kubectl top`, node exporters, and cloud consoles.

## 2. htop screen — what the columns mean

Exact names vary slightly by version/OS. Common ones:

| Column / meter | Meaning |
| -------------- | ------- |
| **CPU%** (per process) | Share of *one* CPU; on 8 cores a process can show up to ~800% if threaded |
| **CPU meters** (header) | Per-core bars: user / system / nice / irq / steal / io-wait (colors) |
| **MEM% / RES** | Resident set — RAM actually backed by physical pages |
| **VIRT** | Virtual address space (can be huge and still fine) |
| **SHR** | Shared memory (libraries, shm) |
| **SWAP** | Pages paged out (pressure signal) |
| **TIME+** | Cumulative CPU time |
| **PRI / NI** | Priority / nice (−20..19; higher nice = nicer = less CPU) |
| **S** | State: `R` running, `S` sleep, `D` uninterruptible (often disk), `Z` zombie, `T` stopped |
| **P** | Last CPU id |
| **Command** | argv |

Keyboard habits worth muscle memory:

- `F2` setup · `F3`/`/` search · `F4` filter · `F5` tree · `F6` sort
- `k` kill · `r` renice · `H` hide userland threads · `K` hide kernel threads
- `p` (some builds) show full path · `t` tree toggle

## 3. Symptom → metric cheat sheet

| Symptom | Likely htop cues | First checks |
| ------- | ---------------- | ------------ |
| Machine sluggish, fans up | One/few PIDs at high CPU%, cores red | Identify Command; is it expected? |
| System thrashing | High SWAP, RES sum ≈ RAM, CPU in wait | Who grew RES? kill/renice or add RAM |
| Disk busy, CPU idle-ish | `D` states, IO-wait in header | `lsof`, iostat; which files? |
| Job “running” but no progress | `S` sleep, low CPU | Waiting on lock/network? hung child? |
| Defunct entries | State `Z`, odd Command | Parent not `wait()`ing — fix parent |

## 4. Companion CLI (same questions, scriptable)

```bash
ps auxww | head
ps -o pid,ppid,ni,state,pcpu,pmem,rss,command -p <pid>
pgrep -af cpu_burn
kill -TERM <pid>          # polite
kill -KILL <pid>          # last resort
renice +10 -p <pid>       # lower priority
lsof -p <pid>
```

Use htop to *discover*, CLI to *automate* and *record*.

## 5. Relation to containers & K8s (preview, not required)

Inside a container you often see only that container’s PIDs (PID namespace).
On the **host**, the same process has a different PID. Node-level htop shows
the noisy neighbor; `kubectl top pod` shows the aggregate. This lab stays on
the host so the mental model is clear before namespaces obscure it.

## 6. Safety

- Prefer `SIGTERM` before `SIGKILL`.
- Do not `kill -9` random system PIDs.
- Lab workloads are tagged under `workloads/*.py` and `.run/*.pid` — stop with
  `mise run workloads-stop`.
