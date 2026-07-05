use std::time::Duration;
use std::fmt::{self, Debug};
use std::rc::Rc;

use tui_logger::*;

use crate::features::app::{Action, Page};
use crate::features::top_page::TopPage;
use crate::env::Env;
use crate::config::get_config;
use crate::features::top_page::TopPageAction;


#[derive(Clone)]
pub struct LogWidgetState(pub Rc<TuiWidgetState>);

impl Debug for LogWidgetState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LogWidgetState({:p})", self.0)
    }
}


#[derive(Clone, Debug)]
pub struct LogPage {
    pub state: LogWidgetState,
}

impl LogPage {
    pub fn new() -> Self {
        Self {
            state: LogWidgetState(Rc::new(TuiWidgetState::default())),
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

pub fn log_page_reducer(state: &mut LogPage, action: &LogPageAction, env: &Env) {
    match action {
        _ => (),
    }
}
