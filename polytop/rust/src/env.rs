use tokio::sync::mpsc;

use crate::event::Event;
use crate::config::{get_config, Config};

pub struct Env {
    pub sender: mpsc::UnboundedSender<Event>,
    pub receiver: mpsc::UnboundedReceiver<Event>,
    pub config: &'static Config,
}

impl Env {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel::<Event>();
        Self {
            sender,
            receiver,
            config: get_config(),
        }
    }
}
