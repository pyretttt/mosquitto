use std::error::Error;
use std::fmt;
use std::mem;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::widgets::TableState;

use crate::pair::Pair;
use crate::features::app::Action;
use crate::event::Event;
use crate::env::Env;

static TOP_PAGE_TITLE: &str = "Polytop";

#[derive(Clone, Debug)]
pub struct TopPage {
    pub left_title: &'static str,
    pub right_title: String,

    pub error_msg: Option<Pair<String, String>>,
    pub current_pane: u32,

    pub status_pane: StatusPane,
    pub markets_pane: MarketsPane,
    pub selected_market_pane: MarketSummary,
    pub chart_activity_pane: ChartActivityPane,
    pub command_popup: CommandPopup,

    pub markets_next_cursor: Option<String>,
    pub markets_load_session: Option<String>,

    pub is_loading: bool,
}

impl Default for TopPage {
    fn default() -> Self {
        Self::mock_data()
    }
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
    pub filter: String,
    pub markets: Vec<Market>,
    pub table_state: TableState,
}

#[derive(Clone, Debug, Default)]
pub struct Market {
    pub title: String,
    pub slug: String,
    pub bookmarked: bool,
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
    pub yes_bid: f64,
    pub yes_ask: f64,
    pub no_bid: f64,
    pub no_ask: f64,
    pub spread: f64,
    pub volume24h: f64,
    pub liquidity: f64,
    pub open_interest: f64,
    pub end_date: String,
}

#[derive(Clone, Debug, Default)]
pub struct ChartActivityPane {
    pub title: &'static str,
    pub chart_lines: Vec<String>,
    pub activities: Vec<ActivityEntry>,
}

#[derive(Clone, Debug, Default)]
pub struct ActivityEntry {
    pub time: String,
    pub label: String,
    pub value: String,
    pub kind: ActivityKind,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActivityKind {
    #[default]
    Muted,
    Positive,
    Negative,
    Accent,
    Warning,
}

#[derive(Clone, Debug, Default)]
pub struct CommandPopup {
    pub filter: String,
    pub sort: String,
    pub mode: String,
}

#[derive(Clone, Debug)]
pub struct MarketLoadResult {
    pub markets: Vec<Market>,
    pub next_cursor: String,
    pub session: String,
}

#[derive(Clone, Debug)]
pub enum TopPageAction {
    SelectPane(u32),
    MarketsLoadRequested,
    MarketsRequestFinished(Result<MarketLoadResult, TopPageError>),
    HideErrorMsg { token: String },
}

impl Into<Event> for TopPageAction {
    fn into(self) -> Event {
        Event::App(Action::TopPage(self))
    }
}

#[derive(Clone, Debug)]
pub enum TopPageError {
    MarketsRequestFailed,
}

pub fn top_page_reducer(top_page: &mut TopPage, action: &mut TopPageAction, env: &Env) {
    match action {
        TopPageAction::SelectPane(pane) => {
            top_page.current_pane = *pane;
        },
        TopPageAction::MarketsLoadRequested => {
            if top_page.markets_load_session.is_some() { return; }

            let current_session = (env.gen_token)();
            top_page.is_loading = true;
            top_page.markets_load_session = Some(current_session.clone());
            let sender: tokio::sync::mpsc::UnboundedSender<Event> = env.sender.clone();
            let polymarket_client = env.polymarket_client.clone();
            let markets_next_cursor = top_page.markets_next_cursor.clone();
            env.fire_and_forget(async move {
                match polymarket_client.markets(markets_next_cursor).await {
                    Ok(markets) => {
                        let markets_data = markets.data.into_iter().map(|market| {
                            let token_price = |outcome: &str| {
                                market.tokens
                                    .iter()
                                    .find(|token| token.outcome.eq_ignore_ascii_case(outcome))
                                    .and_then(|token| token.price.to_string().parse::<f64>().ok())
                                    .map(|price| price * 100.0)
                                    .unwrap_or_default()
                            };
                            let yes_market_price = token_price("Yes");
                            let no_market_price = token_price("No");

                            Market {
                                title: market.question,
                                slug: market.market_slug,
                                bookmarked: false,
                                yes_market_price,
                                no_market_price,
                                volume24h: 0.0,
                                movement: 0.0,
                                spread: 0.0,
                            }
                        });
                        let next_cursor = markets.next_cursor.clone();
                        _ = sender.send(
                            TopPageAction::MarketsRequestFinished(
                                Ok(MarketLoadResult { markets: markets_data.collect(), next_cursor, session: current_session } )
                            ).into()
                        );
                    },
                    Err(err) => {
                        log::error!("MarketsRequestFailed: {:?}", err);
                        _ = sender.send(TopPageAction::MarketsRequestFinished(Err(TopPageError::MarketsRequestFailed)).into());
                    }
                }
            });
        },
        TopPageAction::MarketsRequestFinished(result) => {
            match result {
                Ok(load_result) => {
                    if let Some(ref session) = top_page.markets_load_session && load_result.session.eq(session) {
                        top_page.markets_pane.markets = mem::take(&mut load_result.markets);
                        top_page.markets_next_cursor = Some(mem::take(&mut load_result.next_cursor));
                    }
                },
                Err(_) => {
                    let current_token = (env.gen_token)();
                    let sender = env.sender.clone();
                    top_page.error_msg = Some(Pair::new(
                        "Failed to load markets data, press `R` to retry".to_owned(),
                        current_token.clone()
                    ));
                    let sleep_fn = env.sleep.clone();
                    _ = env.fire_and_forget(async move {
                        sleep_fn.sleep(3000).await;
                        _ = sender.send(TopPageAction::HideErrorMsg { token: current_token }.into());
                    });
                }
            }
            top_page.is_loading = false;
        },
        TopPageAction::HideErrorMsg { token } => {
            if top_page.error_msg.as_ref().map_or(false,|e| e.right.eq(token)) {
                top_page.error_msg = None;
            }
        },
    }
}

impl TopPage {
    pub fn key_input_middleware(&mut self, key_event: &KeyEvent, env: &Env) -> bool {
        match key_event.code {
            KeyCode::Char(x) if ['1', '2', '3'].contains(&x) => {
                self.current_pane = x.to_digit(10).unwrap_or(1);
                true
            },
            KeyCode::Char(x) if ['j', 'k'].contains(&x) => {
                let selected_idx = self.markets_pane.table_state.selected().unwrap_or(0);
                if x == 'j' && self.markets_pane.markets.len() > selected_idx + 1 {
                    self.markets_pane.table_state.select_next();
                    let count = self.markets_pane.markets.len();
                    if self.markets_pane.table_state.selected().map_or(false, |idx| idx >= count - 1) {
                        _ = env.sender.send(TopPageAction::MarketsLoadRequested.into());
                    }
                } else if x == 'k' && selected_idx > 0 {
                    self.markets_pane.table_state.select_previous();
                }
                true
            }
            _ => false,
        }
    }
}

// ================================================
// === Mock data
// ================================================
impl TopPage {
    pub fn mock_data() -> Self {
        Self {
            left_title: TOP_PAGE_TITLE,
            right_title: String::new(),
            current_pane: 1,
            status_pane: StatusPane {
                is_online: true,
                ws_live: true,
                latency: 42,
                refresh_interval: 500,
                mode: "observe".into(),
            },
            markets_pane: MarketsPane {
                title: "Top Markets",
                filter: "all".into(),
                markets: vec![
                    Market {
                        title: "Will BTC hit 100k in 2026?".to_owned(),
                        slug: "btc-100k-2026".into(),
                        bookmarked: true,
                        yes_market_price: 63.0,
                        no_market_price: 38.0,
                        volume24h: 842_000.0,
                        movement: 4.0,
                        spread: 2.0,
                    },
                    Market {
                        title: "Fed cuts rates by Sep?".to_owned(),
                        slug: "fed-cuts-sep".into(),
                        bookmarked: false,
                        yes_market_price: 41.0,
                        no_market_price: 60.0,
                        volume24h: 611_000.0,
                        movement: -2.0,
                        spread: 3.0,
                    },
                    Market {
                        title: "Lakers win tonight?".to_owned(),
                        slug: "lakers-win-tonight".into(),
                        bookmarked: false,
                        yes_market_price: 55.0,
                        no_market_price: 46.0,
                        volume24h: 570_000.0,
                        movement: 1.0,
                        spread: 2.0,
                    },
                    Market {
                        title: "ETH ETF inflows above $1B?".to_owned(),
                        slug: "eth-etf-inflows-1b".into(),
                        bookmarked: false,
                        yes_market_price: 72.0,
                        no_market_price: 29.0,
                        volume24h: 510_000.0,
                        movement: 6.0,
                        spread: 4.0,
                    },
                    Market {
                        title: "Trump wins popular vote?".to_owned(),
                        slug: "trump-popular-vote".into(),
                        bookmarked: true,
                        yes_market_price: 49.0,
                        no_market_price: 52.0,
                        volume24h: 421_000.0,
                        movement: -1.0,
                        spread: 2.0,
                    },
                    Market {
                        title: "CPI below forecast?".to_owned(),
                        slug: "cpi-below-forecast".into(),
                        bookmarked: false,
                        yes_market_price: 36.0,
                        no_market_price: 65.0,
                        volume24h: 390_000.0,
                        movement: -5.0,
                        spread: 5.0,
                    },
                    Market {
                        title: "SpaceX launch this week?".to_owned(),
                        slug: "spacex-launch-week".into(),
                        bookmarked: false,
                        yes_market_price: 83.0,
                        no_market_price: 18.0,
                        volume24h: 311_000.0,
                        movement: 8.0,
                        spread: 3.0,
                    },
                    Market {
                        title: "Oil closes above $90?".to_owned(),
                        slug: "oil-above-90".into(),
                        bookmarked: false,
                        yes_market_price: 22.0,
                        no_market_price: 79.0,
                        volume24h: 280_000.0,
                        movement: -3.0,
                        spread: 4.0,
                    },
                ],
                table_state: TableState::default().with_selected(Some(0)),
            },
            selected_market_pane: MarketSummary {
                title: "Will BTC hit 100k in 2026?",
                slug: "btc-100k-2026".into(),
                yes_market_price: 63.0,
                no_market_price: 38.0,
                yes_bid: 62.0,
                yes_ask: 64.0,
                no_bid: 37.0,
                no_ask: 39.0,
                spread: 2.0,
                volume24h: 842_100.0,
                liquidity: 184_300.0,
                open_interest: 2_400_000.0,
                end_date: "2026-12-31".into(),
            },
            chart_activity_pane: ChartActivityPane {
                title: "Chart + Activity",
                chart_lines: vec![
                    " 70¢ ┤                     ╭╮".into(),
                    " 65¢ ┤              ╭──────╯╰─╮".into(),
                    " 60¢ ┤      ╭───────╯         ╰╮".into(),
                    " 55¢ ┤ ╭────╯                  ╰─".into(),
                    "     └────────────────────────────".into(),
                ],
                activities: vec![
                    ActivityEntry {
                        time: "17:42".into(),
                        label: "price".into(),
                        value: "+4¢".into(),
                        kind: ActivityKind::Positive,
                    },
                    ActivityEntry {
                        time: "17:41".into(),
                        label: "best bid".into(),
                        value: "62¢".into(),
                        kind: ActivityKind::Accent,
                    },
                    ActivityEntry {
                        time: "17:40".into(),
                        label: "trade".into(),
                        value: "219 @45¢".into(),
                        kind: ActivityKind::Warning,
                    },
                    ActivityEntry {
                        time: "17:39".into(),
                        label: "spread".into(),
                        value: "2¢".into(),
                        kind: ActivityKind::Muted,
                    },
                ],
            },
            command_popup: CommandPopup {
                filter: "politics volume>100k".into(),
                sort: "move".into(),
                mode: "NORMAL".into(),
            },
            error_msg: None,
            markets_next_cursor: None,
            markets_load_session: None,
            is_loading: false,
        }
    }
}

impl fmt::Display for TopPageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{:?}", self)
    }
}

impl Error for TopPageError {}