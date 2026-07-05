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
pub struct LoadingPage {
    pub progress: f32,
    pub throbbler_state: throbber_widgets_tui::ThrobberState,
    pub loading_tip: &'static str,
    pub throbbler_caption: String,
    pub logo_color_index: usize,
    pub is_finished: bool,
    loading_tip_index: usize,
    tick: u16,
}

impl LoadingPage {
    pub fn new(throbbler_caption: impl Into<String>) -> Self {
        Self {
            progress: 0.0,
            loading_tip: LOADING_TIPS[0],
            throbbler_state: throbber_widgets_tui::ThrobberState::default(),
            throbbler_caption: throbbler_caption.into(),
            logo_color_index: 0,
            loading_tip_index: 0,
            tick: 0,
            is_finished: false,
        }
    }

    pub fn tick(&mut self, env: &Env) {
        self.tick = self.tick.wrapping_add(1);
        if self.tick % 3 == 0 {
            self.throbbler_state.calc_next();
            self.logo_color_index = self.logo_color_index.wrapping_add(1);
        }
        if self.tick % (get_config().tick_rate * 2.0) as u16 == 0 {
            self.loading_tip_index = (self.loading_tip_index + 1) % LOADING_TIPS.len();
            self.loading_tip = LOADING_TIPS[self.loading_tip_index];
        }
        if !self.is_finished && self.tick % 10 == 0 {
            let rng = (env.rng)(Some(0.01..0.1));
            self.progress = (self.progress + rng).min(MAX_FAKE_PROGRESS);
        }
    }
}

impl Default for LoadingPage {
    fn default() -> Self {
        Self::new(THROBBLER_CAPTION)
    }
}

#[derive(Clone, Debug)]
pub enum LoadingPageAction {
    Finished,
}

pub fn loading_page_reducer(state: &mut LoadingPage, action: &LoadingPageAction, env: &Env) {
    match action {
        LoadingPageAction::Finished => {
            state.is_finished = true;
            state.progress = 1.0;
            let sender = env.sender.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(300)).await;
                _ = sender.send(
                    Action::OpenTopPage(
                        TopPage::mock_data()
                    ).into()
                );
                _ = sender.send(
                    TopPageAction::MarketsLoadRequested.into()
                );

            });
        }
    }
}
