use futures::channel::mpsc;
use crate::models::app_state::Action;

pub enum Event {
    Tick,
    App(Action),
}

pub struct Env {
    pub event_tx: mpsc::UnboundedSender<Event>,
    pub event_rx: mpsc::UnboundedReceiver<Event>,
}

impl Env {
    pub fn new() -> Self {
        let (event_tx, event_rx) = mpsc::unbounded();
        Self {
            event_tx,
            event_rx,
        }
    }
}