use std::time::Duration;

use futures::{FutureExt, StreamExt};
use tokio::sync::mpsc;
use crossterm::event::Event as CrosstermEvent;

use crate::models::app_state::Action;
use crate::config::config;

pub enum Event {
    Tick,
    Crossterm(CrosstermEvent),
    App(Action),
}

pub struct EventLoop {
    /// Event sender channel.
    pub sender: mpsc::UnboundedSender<Event>,
    /// Event receiver channel.
    pub receiver: mpsc::UnboundedReceiver<Event>,
}

impl EventLoop {
    pub fn new(tx: mpsc::UnboundedSender<Event>, rx: mpsc::UnboundedReceiver<Event>) -> Self {
        Self { sender: tx, receiver: rx }
    }

    pub async fn next(&mut self) -> color_eyre::Result<Event> {
        self.receiver
            .recv()
            .await
            .ok_or(color_eyre::eyre::eyre!("Failed to receive event"))
    }
}

pub async fn run(sender: mpsc::UnboundedSender<Event>) -> color_eyre::Result<()> {
    let mut reader = crossterm::event::EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_secs_f64(1.0 / config().tick_rate));
    loop {
        let tick_delay = tick.tick();
        let crossterm_event = reader.next().fuse();

        tokio::select! {
            _ = sender.closed() => {
                break;
            }
            _ = tick_delay => {
                let _ = sender.send(Event::Tick);
            }
            Some(Ok(evt)) = crossterm_event => {
                let _ = sender.send(Event::Crossterm(evt));
            }
        }
    }
    Ok(())
}
