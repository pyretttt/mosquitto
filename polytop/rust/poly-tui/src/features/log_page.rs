use std::time::Duration;

use crate::features::app::{Action, Page};
use crate::features::top_page::TopPage;
use crate::env::Env;
use crate::config::get_config;
use crate::features::top_page::TopPageAction;

static LOADING_TIPS: [&str; 4] = [
    "Press `q` to quit",
    "Press `/` to open command palette",
    "Press `?` to open help",
    "Press `Ctrl+c` to quit",
];

static MAX_FAKE_PROGRESS: f32 = 0.87;
static THROBBLER_CAPTION: &str = "Cooking smooth performance...";

#[derive(Clone, Debug)]
pub struct LogPage {
}

impl LogPage {
    pub fn new() -> Self {
        Self {
        }
    }

}

impl Default for LogPage {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub enum LoadingPageAction {
    Finished,
}

pub fn loading_page_reducer(state: &mut LogPage, action: &LogPageAction, env: &Env) {
    match action {
    }
}
