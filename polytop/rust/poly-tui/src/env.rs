use std::ops::Range;
use std::future::Future;

use tokio::task::JoinHandle;
use tokio::sync::mpsc;
use uuid::Uuid;
use rand::RngExt;
use poly_core::api::PolymarketClient;

use crate::event::Event;
use crate::config::{get_config, Config};

pub struct Env {
    pub sender: mpsc::UnboundedSender<Event>,
    pub receiver: mpsc::UnboundedReceiver<Event>,
    pub config: &'static Config,
    pub gen_token: Box<dyn Fn() -> String + 'static + Send + Sync>,
    pub rng: Box<dyn Fn(Option<Range<f32>>) -> f32 + 'static + Send + Sync>,
    pub polymarket_client: PolymarketClient,
}

impl Env {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel::<Event>();
        Self {
            sender,
            receiver,
            config: get_config(),
            gen_token: Box::new(|| Uuid::new_v4().to_string()),
            rng: Box::new(|range| rand::rng().random_range(range.unwrap_or(0.0..1.0))),
            polymarket_client: PolymarketClient::default(),
        }
    }

    pub fn fire_and_forget<F: Future + Send + 'static>(&self, future: F) -> JoinHandle<F::Output>
        where F::Output: Send + 'static {
        tokio::spawn(future)
    }
}
