use tokio::sync::mpsc;
use uuid::Uuid;

use crate::event::Event;
use crate::config::{get_config, Config};
use tokio::task::JoinHandle;
use std::future::Future;

pub struct Env {
    pub sender: mpsc::UnboundedSender<Event>,
    pub receiver: mpsc::UnboundedReceiver<Event>,
    pub config: &'static Config,
    pub gen_token: Box<dyn Fn() -> String + 'static + Send + Sync>,
}

impl Env {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel::<Event>();
        Self {
            sender,
            receiver,
            config: get_config(),
            gen_token: Box::new(|| Uuid::new_v4().to_string()),
        }
    }

    pub fn fire_and_forget<F: Future + Send + 'static>(&self, future: F) -> JoinHandle<F::Output>
        where F::Output: Send + 'static {
        tokio::spawn(future)
    }
}
