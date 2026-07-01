use crossterm::event::{KeyCode, KeyEvent};

use crate::env::Env;

static TOP_PAGE_TITLE: &str = "Polytop";

#[derive(Clone, Debug, Default)]
pub struct TopPage {
    pub left_title: &'static str,
    pub right_title: String,

    pub current_pane: u32,

    pub status_pane: StatusPane,
    pub markets_pane: MarketsPane,
    pub selected_market_pane: MarketSummary,
    pub chart_activity_pane: ChartActivityPane,
}

#[derive(Clone, Debug, Default)]
pub struct StatusPane {
    pub is_online: bool,
    pub ws_live: bool,
    pub latency: u64,
    pub refresh_interval: u64,
    pub mode: String,
}

#[derive(Clone, Debug, Default)]
pub struct MarketsPane {
    pub title: &'static str,
    pub markets: Vec<Market>,
}

#[derive(Clone, Debug, Default)]
pub struct Market {
    pub title: &'static str,
    pub slug: String,
    pub yes_market_price: f64,
    pub no_market_price: f64,
    pub volume24h: f64,
    pub movement: f64,
    pub spread: f64,
}

#[derive(Clone, Debug, Default)]
pub struct MarketSummary {
    pub title: &'static str,
    pub slug: String,
    pub yes_market_price: f64,
    pub no_market_price: f64,
    pub volume24h: f64,
    pub movement: f64,
    pub spread: f64,
}

#[derive(Clone, Debug, Default)]
pub struct ChartActivityPane {
    pub title: &'static str,
}

#[derive(Clone, Debug)]
pub enum TopPageAction {
    SelectPane(u32),
}

pub fn top_page_reducer(top_page: &mut TopPage, action: &TopPageAction, env: &Env) {
    match action {
        TopPageAction::SelectPane(pane) => {
            top_page.current_pane = *pane;
        }
    }
}

impl TopPage {
    pub fn key_input_middleware(&mut self, key_event: &KeyEvent, _env: &Env) -> bool {
        match key_event.code {
            KeyCode::Char(x) if ['1', '2', '3'].contains(&x) => {
                self.current_pane = x.to_digit(10).unwrap_or(1);
                true
            }
            _ => false
        }
    }
}