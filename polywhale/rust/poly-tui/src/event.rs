use crossterm::event::Event as CrosstermEvent;
use futures::{FutureExt, StreamExt};
use std::time::Duration;
use tokio::sync::mpsc;

use crate::features::app::Action;
use crate::config::get_config;

/// Representation of all possible events.
#[derive(Clone, Debug)]
pub enum Event {
    Tick,
    Crossterm(CrosstermEvent),
    App(Action),
}

pub async fn sidecar_event_loop(sender: mpsc::UnboundedSender<Event>) -> color_eyre::Result<()> {
    let mut reader = crossterm::event::EventStream::new();
    let tick_rate = Duration::from_secs_f64(1.0 / get_config().tick_rate);
    let mut tick = tokio::time::interval(tick_rate);

    loop {
        tokio::select! {
            _ = tick.tick() => {
                let _ = sender.send(Event::Tick)?;
            }
            Some(Ok(evt)) = reader.next().fuse() => {
                let _ = sender.send(Event::Crossterm(evt))?;
            }
            _ = sender.closed() => {
                break;
            }
        }
    }
    Ok(())
}