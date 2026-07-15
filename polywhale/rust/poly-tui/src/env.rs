use std::ops::Range;
use std::future::Future;
use std::time::Duration;

use tokio::task::JoinHandle;
use tokio::sync::mpsc;
use uuid::Uuid;
use rand::RngExt;
use ratatui::prelude::Size;
use poly_core::client::PolymarketClient;

use crate::event::Event;
use crate::config::{get_config, Config};
use crate::top_page_service::TopPageService;

#[derive(Clone, Debug)]
pub struct SleepFn {}

impl SleepFn {
    pub async fn sleep(&self, milliseconds: u64) {
        tokio::time::sleep(Duration::from_millis(milliseconds)).await;
    }
}

impl Default for SleepFn {
    fn default() -> Self {
        Self {}
    }
}

pub struct Env {
    pub sender: mpsc::UnboundedSender<Event>,
    pub receiver: mpsc::UnboundedReceiver<Event>,
    pub config: &'static Config,
    pub gen_token: Box<dyn Fn() -> String + 'static + Send + Sync>,
    pub rng: Box<dyn Fn(Option<Range<f32>>) -> f32 + 'static + Send + Sync>,
    pub polymarket_client: PolymarketClient,
    pub sleep: SleepFn,
    pub top_page_svc: TopPageService,
    pub ui: UI,
}

pub struct UI {
    pub window_size: Size,
}

impl Env {
    pub fn new(window_size: Size) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel::<Event>();

        let polymarket_client = PolymarketClient::default();
        Self {
            sender,
            receiver,
            config: get_config(),
            gen_token: Box::new(|| Uuid::new_v4().to_string()),
            rng: Box::new(|range| rand::rng().random_range(range.unwrap_or(0.0..1.0))),
            polymarket_client: PolymarketClient::default(),
            sleep: SleepFn::default(),
            top_page_svc: TopPageService::new(polymarket_client),
            ui: UI {
                window_size: window_size,
            },
        }
    }

    pub fn fire_and_forget<F: Future + Send + 'static>(&self, future: F) -> JoinHandle<F::Output>
        where F::Output: Send + 'static {
        tokio::spawn(future)
    }
}
