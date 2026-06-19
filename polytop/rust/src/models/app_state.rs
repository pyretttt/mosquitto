use crate::event_loop::{Event};
use crate::env::Env;

#[derive(Clone, Debug)]
pub struct AppState {
    pub page: Page,
    pub counter: i32,
}

#[derive(Clone, Debug)]
pub enum Page {
    Intro(IntroPage),
}

#[derive(Clone, Debug)]
pub struct IntroPage {
    pub title: String,
    pub text: String,
}

pub fn app_state_reduce(app_state: &mut AppState, action: Action, _env: &Env) {
    match action {
        Action::Next => app_state.counter += 1,
    }
}

pub enum Action {
    Next,
}

impl From<Action> for Event {
    fn from(value: Action) -> Self {
        Event::App(value)
    }
}
