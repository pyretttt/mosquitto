use crate::env::Env;

use super::action::WindowSizeAction;
use super::state::WindowSize;

pub fn window_size_reducer(_state: &mut WindowSize, action: &WindowSizeAction, _env: &Env) {
    match action {
        _ => (),
    }
}
