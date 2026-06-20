use std::time::Duration;

use uuid::Uuid;
use crossterm::event::{KeyCode, KeyEventKind, KeyModifiers};

use crate::env::Env;
use crate::event::{Event};

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
        Event::Tick => (),
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
            page: Page::Intro(IntroPage {
                title: "Polytop - Polymarket Monitor".to_owned(),
                text: "Hello, world!".to_owned(),
            }),
            counter: 0,
            running: true,
            increment_token: None,
        }
    }
}
