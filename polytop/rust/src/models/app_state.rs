use tokio::sync::mpsc;
use crate::event_loop::{EventLoop, Event};

// AppState

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

impl AppState {
    pub fn reduce(&mut self, action: Action, env: &Env) {
        match action {
            Action::Next => app_state.counter += 1,
        }    
    }
}

// Environment
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

// Actions
pub enum Action {
    Next,
}

impl Into<Event> for Action {
    fn into(self) -> Event { Event::App(self) }
}