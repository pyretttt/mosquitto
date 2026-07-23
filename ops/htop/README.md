# Htop & process management

A hands-on lab for **htop**, process metrics, and the companion CLI tools you
reach for when a shared DevOps / MLOps box is noisy: pegged CPUs, climbing RSS,
IO wait, hung workers, and fork trees.

> This is a **scaffold**. Most pieces are intentionally left as exercises.
> Look for `TODO(you)` markers and the checklists in `TASKS.md`.

## Learning outcomes

- Install a reproducible tool chain with mise and get htop running
- Read htop meters/columns and map them to CPU, memory, IO, and process state
- Diagnose synthetic CPU / memory / IO / hang / zombie scenarios
- Intervene safely with filter, tree view, renice, signals, and `lsof`
- Relate host process views to MLOps/DevOps situations (training jobs, loaders, stuck workers)

## Components

| Concern | Path | Backed by |
| ------- | ---- | --------- |
| Tooling / tasks | `mise.toml` | mise + system htop |
| Concepts | `docs/CONCEPTS.md` | markdown |
| Student notes | `docs/notes.md` | markdown |
| Synthetic workloads | `workloads/` | Python 3 |
| Start/stop helpers | `scripts/` | bash |

## Layout

```
htop/
├── README.md
├── TASKS.md
├── mise.toml
├── .gitignore
├── docs/
│   ├── CONCEPTS.md
│   └── notes.md
├── workloads/
│   ├── cpu_burn.py
│   ├── mem_hog.py
│   ├── io_churn.py
│   ├── fork_tree.py
│   └── hung_worker.py
└── scripts/
    ├── bootstrap.sh
    ├── workloads.sh
    └── launch-htop.sh
```

## Prerequisites

- [mise](https://mise.jdx.dev/)
- htop via your OS package manager (`brew install htop` or `apt install htop`)
- macOS or Linux (process tools differ slightly; notes call that out)
- ~4–6 hours across 1–2 days

## Quick start

```bash
cd ops/htop
mise install
# install htop if needed: brew install htop
mise run bootstrap
mise run verify-tools
mise run workloads-start
mise run htop
# when finished:
mise run workloads-stop
```

## Where to work

Start with `TASKS.md`. Skim `docs/CONCEPTS.md` alongside each section.
