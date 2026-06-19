use std::time::Duration;
use std::env;

use color_eyre::eyre::OptionExt;
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
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel::<Event>();
        Self { sender, receiver }
    }

    pub fn run_(&mut self) -> color_eyre::Result<()> {
        let mut reader = crossterm::event::EventStream::new();
        let mut tick = tokio::time::interval(Duration::from_secs_f64(1.0 / config().tick_rate));
        loop {
            let tick_delay = tick.tick();
            let crossterm_event = reader..fuse();

            tokio::select! {
                _ = self.sender.closed() => {
                    break;
                }
                _ = tick_delay => {
                    self.sender.send(Event::Tick);
                }
                Some(Ok(evt)) = crossterm_event => {
                    self.sender.send(Event::Crossterm(evt));
                }
            }
        }
        Ok(())
    }
}

