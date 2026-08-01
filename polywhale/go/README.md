# polywhale (Go)

Go rewrite of the Rust `poly-tui` Polymarket dashboard, built with
[Bubble Tea](https://github.com/charmbracelet/bubbletea) and
[Lip Gloss](https://github.com/charmbracelet/lipgloss).

## Requirements

- [mise](https://mise.jdx.dev/) — manages the Go toolchain and the
  environment (`TICK_RATE`), and provides the task runner.

## Usage

```bash
mise install       # install the pinned Go toolchain
mise run tidy      # fetch dependencies
mise run run       # start the TUI
```

Other tasks:

| Task              | Description                |
| ----------------- | -------------------------- |
| `mise run build`  | Build `bin/polywhale`      |
| `mise run fmt`    | Format sources             |
| `mise run vet`    | Static analysis            |
| `mise run test`   | Run tests                  |
| `mise run check`  | fmt + vet + build          |

## Environment

- `TICK_RATE` — UI ticks per second (default `30`, set in `.mise.toml`).

## Keybindings

- Loading page: `enter` finishes loading and opens the dashboard.
- Dashboard: `s` search, `1-3` focus pane, `j/k` move, `→`/`enter` focus
  markets, `←`/`esc` back to events.
- `/` opens the command palette (`help`, `quit`, `intro`, `log`).
- `q`, `esc` or `ctrl+c` quits.

Logs are written to `$TMPDIR/polywhale.log` (rotated at 1 MiB) and are
viewable in-app via the `log` command.
