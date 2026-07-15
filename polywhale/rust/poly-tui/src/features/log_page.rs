use std::fmt::{self, Debug};
use std::rc::Rc;

use tui_logger::*;
use crossterm::event::{KeyCode};

use crate::env::Env;
use crate::features::app::Action;

#[derive(Clone)]
pub struct LogWidgetState(pub Rc<TuiWidgetState>);

impl Debug for LogWidgetState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LogWidgetState({:p})", self.0)
    }
}


#[derive(Clone, Debug)]
pub struct LogPage {
    pub logs_state: LogWidgetState,
    pub applied_app_logs_filter: bool,
    pub help: [&'static str; 3],
}

impl LogPage {
    pub fn new() -> Self {
        Self {
            logs_state: LogWidgetState(Rc::new(TuiWidgetState::default())),
            applied_app_logs_filter: false,
            help: [
                "Q: Quit | Tab: Switch state | ↑/↓: Select target | f: Focus target | s: Switch app logs filter",
                "←/→: Display level | +/-: Filter level | Space: Toggle hidden targets",
                "h: Hide target selector | j/k: Scroll | Esc: Cancel scroll",
            ]
        }
    }

}

impl Default for LogPage {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub enum LogPageAction {
}

pub fn log_page_reducer(_state: &mut LogPage, action: &LogPageAction, _env: &Env) {
    match action {
        _ => (),
    }
}

impl LogPage {
    pub fn key_input_middleware(
        &mut self,
        key_event: &mut crossterm::event::KeyEvent,
        env: &Env,
    ) -> bool{
        match key_event.code {
            KeyCode::Tab => (),
            KeyCode::Char(' ') => self.logs_state.0.transition(TuiWidgetEvent::SpaceKey),
            KeyCode::Esc => self.logs_state.0.transition(TuiWidgetEvent::EscapeKey),
            KeyCode::Char('k') => self.logs_state.0.transition(TuiWidgetEvent::PrevPageKey),
            KeyCode::Char('j') => self.logs_state.0.transition(TuiWidgetEvent::NextPageKey),
            KeyCode::Up => self.logs_state.0.transition(TuiWidgetEvent::UpKey),
            KeyCode::Down => self.logs_state.0.transition(TuiWidgetEvent::DownKey),
            KeyCode::Left => self.logs_state.0.transition(TuiWidgetEvent::LeftKey),
            KeyCode::Right => self.logs_state.0.transition(TuiWidgetEvent::RightKey),
            KeyCode::Char('+') => self.logs_state.0.transition(TuiWidgetEvent::PlusKey),
            KeyCode::Char('-') => self.logs_state.0.transition(TuiWidgetEvent::MinusKey),
            KeyCode::Char('h') => self.logs_state.0.transition(TuiWidgetEvent::HideKey),
            KeyCode::Char('f') => self.logs_state.0.transition(TuiWidgetEvent::FocusKey),
            KeyCode::Char('q') => {
                _ = env.sender.send(Action::CloseOverlay.into());
            },
            KeyCode::Char('s') => {
                self.applied_app_logs_filter = !self.applied_app_logs_filter;
                match Rc::get_mut(&mut self.logs_state.0) {
                    Some(state) => {
                        if self.applied_app_logs_filter {
                            *state = TuiWidgetState::default()
                                .set_default_display_level(LevelFilter::Off)
                                .set_level_for_target("app", LevelFilter::Trace)
                        } else {
                            *state = TuiWidgetState::default();
                        }
                    },
                    None => log::error!(target: "app", "[LogPage] Failed to get logs state, during switching filter"),
                }

            },
            _ => return false,
        }
        true
    }
}