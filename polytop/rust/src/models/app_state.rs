use std::time::Duration;

use uuid::Uuid;
use crossterm::event::{KeyCode, KeyEventKind, KeyModifiers};

use crate::env::Env;
use crate::event::{Event};
use crate::config::get_config;

#[derive(Clone, Debug)]
pub struct AppState {
    pub page: Page,
    pub counter: i32,
    pub running: bool,
    pub increment_token: Option<String>,
}

#[derive(Clone, Debug)]
pub enum Page {
    Intro(IntroPage),
    LoadingPage(LoadingPage),
    Main(MainPage),
}

#[derive(Clone, Debug)]
pub struct IntroPage {
    pub title: String,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct MainPage {
    pub title: String,
    pub text: String,
}

const LOADING_TIPS: [&str; 4] = [
    "Press `q` to quit",
    "Press `/` to open command palette",
    "Press `?` to open help",
    "Press `Ctrl+c` to quit",
];

#[derive(Clone, Debug)]
pub struct LoadingPage {
    pub progress: f32,
    pub throbbler_state: throbber_widgets_tui::ThrobberState,
    pub loading_tip: &'static str,
    pub throbbler_caption: String,
    pub logo_color_index: usize,
    loading_tip_index: usize,
    loading_tip_tick: u8,
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
            loading_tip_tick: 0,
        }
    }

    pub fn tick(&mut self) {
        self.loading_tip_tick = self.loading_tip_tick.wrapping_add(1);
        if self.loading_tip_tick % 3 == 0 {
            self.throbbler_state.calc_next();
            self.logo_color_index = self.logo_color_index.wrapping_add(1);
        }
        if self.loading_tip_tick % (get_config().tick_rate * 2.0) as u8 == 0 {
            self.loading_tip_index = (self.loading_tip_index + 1) % LOADING_TIPS.len();
            self.loading_tip = LOADING_TIPS[self.loading_tip_index];
        }
    }
}

impl Default for LoadingPage {
    fn default() -> Self {
        Self::new("Preparing smooth performance...")
    }
}

#[derive(Clone, Debug)]
pub enum Action {
    Next(String),
    Quit,
}

pub fn app_state_reduce(app_state: &mut AppState, action: &Action) {
    match action {
        Action::Next(ref token) if app_state.increment_token.as_ref() == Some(token) => {
            app_state.counter += 1
        },
        Action::Next(_) => (),
        Action::Quit => app_state.running = false,
    }
}

pub fn app_reducer(app_state: &mut AppState, event: &mut Event, env: &mut Env) {
    match event {
        Event::Tick => {
            if let Page::LoadingPage(ref mut loading) = app_state.page {
                loading.tick();
            }
        }
        Event::Crossterm(crossterm_event) => {
            if let crossterm::event::Event::Key(key_event) = crossterm_event {
                if key_event.kind == KeyEventKind::Press {
                    match key_event.code {
                        KeyCode::Esc | KeyCode::Char('q') => app_state.running = false,
                        KeyCode::Char('c' | 'C') if key_event.modifiers == KeyModifiers::CONTROL => {
                            app_state.running = false
                        }
                        KeyCode::Enter => {
                            // Debounce example
                            let token = Uuid::new_v4().to_string();
                            app_state.increment_token = Some(token.clone());
                            let sender = env.sender.clone();
                            tokio::spawn(async move {
                                tokio::time::sleep(Duration::from_secs(1)).await;
                                sender.send(Event::App(Action::Next(token)))
                            });
                        },
                        _ => {}
                    }
                }
            }
        }
        Event::App(action) => app_state_reduce(app_state, action),
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            page: Page::LoadingPage(LoadingPage::default()),
            counter: 0,
            running: true,
            increment_token: None,
        }
    }
}
