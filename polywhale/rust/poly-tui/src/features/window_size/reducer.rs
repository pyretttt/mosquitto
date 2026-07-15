use crate::env::Env;

use super::action::WindowSizeAction;
use super::state::WindowSize;

pub fn window_size_reducer(state: &mut WindowSize, action: &WindowSizeAction, _env: &Env) {
    match action {
        WindowSizeAction::Resize(size) => {
            state.update_labels(*size);
        },
    }
}
