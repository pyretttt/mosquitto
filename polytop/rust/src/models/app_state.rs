use tokio::sync::mpsc;
use crate::event_loop::{EventLoop, Event};

#[derive(Clone)]
pub struct AppState {
    pub page: Page,
    pub counter: i32,
}


pub struct Env {
    pub event_tx: mpsc::UnboundedSender<Event>,
    pub event_loop: EventLoop,
}

impl Env {
    pub fn new() -> Self {
        let (event_tx, event_rx) = mpsc::unbounded_channel::<Event>();
        let event_loop = EventLoop::new(event_tx.clone(), event_rx);
        Self {
            event_tx,
            event_loop
        }
    }
}

pub fn app_state_reduce(app_state: &mut AppState, action: Action, env: &Env) {
    match action {
        Action::Next => app_state.counter += 1,
    }
}

#[derive(Clone)]
pub enum Page {
    Intro(IntroPage),
}

#[derive(Clone)]
pub struct IntroPage {
    pub title: String,
    pub text: String,
}

pub enum Action {
    Next,
}

impl Into<Event> for Action {
    fn into(self) -> Event { Event::App(self) }
}