use std::future::Future;
use std::ops::Range;
use std::time::Duration;

use poly_core::client::PolymarketClient;
use rand::RngExt;
use ratatui::prelude::Size;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::config::{Config, get_config};
use crate::event::Event;
use crate::top_page_service::TopPageService;

#[derive(Clone, Debug, Default)]
pub struct SleepFn {}

impl SleepFn {
    pub async fn sleep(&self, milliseconds: u64) {
        tokio::time::sleep(Duration::from_millis(milliseconds)).await;
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
    pub required_window_size: Size,
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
                required_window_size: Size::new(120, 40),
            },
        }
    }

    pub fn fire_and_forget<F: Future + Send + 'static>(&self, future: F) -> JoinHandle<F::Output>
    where
        F::Output: Send + 'static,
    {
        tokio::spawn(future)
    }
}
