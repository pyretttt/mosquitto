#!/usr/bin/env bash
# Scaffold a poly-tui feature module + UI component.
#
# Usage:
#   ./scripts/new_feature.sh <snake_case_name>
#
# Creates:
#   src/features/<name>/{mod,state,action,reducer}.rs
#   src/ui_components/<name>_ui.rs
# and registers both in their mod.rs files.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FEATURES_DIR="$ROOT/src/features"
UI_DIR="$ROOT/src/ui_components"

usage() {
  echo "Usage: $0 <snake_case_name>" >&2
  echo "Example: $0 settings_page" >&2
  exit 1
}

[[ $# -eq 1 ]] || usage

NAME="$1"

if [[ ! "$NAME" =~ ^[a-z][a-z0-9_]*$ ]]; then
  echo "Error: name must be snake_case (e.g. settings_page)" >&2
  exit 1
fi

# snake_case -> PascalCase
to_pascal() {
  local s="$1" out="" part
  IFS='_' read -ra parts <<< "$s"
  for part in "${parts[@]}"; do
    out+="$(tr '[:lower:]' '[:upper:]' <<< "${part:0:1}")${part:1}"
  done
  printf '%s' "$out"
}

PASCAL="$(to_pascal "$NAME")"
STATE="${PASCAL}"
ACTION="${PASCAL}Action"
REDUCER="${NAME}_reducer"
DRAW="draw_${NAME}"
FEATURE_DIR="$FEATURES_DIR/$NAME"
UI_FILE="$UI_DIR/${NAME}_ui.rs"

if [[ -e "$FEATURE_DIR" || -e "$FEATURES_DIR/${NAME}.rs" ]]; then
  echo "Error: feature already exists: $NAME" >&2
  exit 1
fi

if [[ -e "$UI_FILE" ]]; then
  echo "Error: UI component already exists: ${NAME}_ui.rs" >&2
  exit 1
fi

mkdir -p "$FEATURE_DIR"

cat > "$FEATURE_DIR/mod.rs" <<EOF
mod action;
mod reducer;
mod state;

pub use action::${ACTION};
pub use reducer::${REDUCER};
pub use state::${STATE};
EOF

cat > "$FEATURE_DIR/state.rs" <<EOF
#[derive(Clone, Debug, Default)]
pub struct ${STATE} {
}
EOF

cat > "$FEATURE_DIR/action.rs" <<EOF
#[derive(Clone, Debug)]
pub enum ${ACTION} {
}

// After adding \`Action::${PASCAL}(${ACTION})\` in features/app.rs:
//
// use crate::event::Event;
// use crate::features::app::Action;
//
// impl Into<Event> for ${ACTION} {
//     fn into(self) -> Event {
//         Event::App(Action::${PASCAL}(self))
//     }
// }
EOF

cat > "$FEATURE_DIR/reducer.rs" <<EOF
use crate::env::Env;

use super::action::${ACTION};
use super::state::${STATE};

pub fn ${REDUCER}(_state: &mut ${STATE}, action: &${ACTION}, _env: &Env) {
    match action {
        _ => (),
    }
}
EOF

cat > "$UI_FILE" <<EOF
use ratatui::Frame;

use crate::features::${NAME}::${STATE};

pub fn ${DRAW}(frame: &mut Frame, state: &${STATE}) {
    let _ = (frame, state);
    // TODO: draw ${STATE}
}
EOF

# Append mod declarations if missing (ensure file ends with a newline first)
append_mod() {
  local file="$1"
  local line="$2"
  if grep -qxF "$line" "$file"; then
    return 0
  fi
  if [[ -s "$file" ]] && [[ "$(tail -c1 "$file" | wc -l)" -eq 0 ]]; then
    printf '\n' >> "$file"
  fi
  printf '%s\n' "$line" >> "$file"
}

append_mod "$FEATURES_DIR/mod.rs" "pub mod ${NAME};"
append_mod "$UI_DIR/mod.rs" "pub mod ${NAME}_ui;"

echo "Created feature module:"
echo "  $FEATURE_DIR/{mod,state,action,reducer}.rs"
echo "  $UI_FILE"
echo
echo "Registered in:"
echo "  src/features/mod.rs"
echo "  src/ui_components/mod.rs"
echo
echo "Next steps (wire into app):"
echo "  1. Add Action::${PASCAL}(${ACTION}) to features/app.rs"
echo "  2. Dispatch ${REDUCER} from app_state_reduce"
echo "  3. Call ${DRAW} from ui.rs when rendering this page"
