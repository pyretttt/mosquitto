use crossterm::event::{KeyCode, KeyEvent};

use crate::env::Env;

static TOP_PAGE_TITLE: &str = "Polytop";

#[derive(Clone, Debug)]
pub struct TopPage {
    pub left_title: &'static str,
    pub right_title: String,

    pub current_pane: u32,

    pub status_pane: StatusPane,
    pub markets_pane: MarketsPane,
    pub selected_market_pane: MarketSummary,
    pub chart_activity_pane: ChartActivityPane,
    pub command_popup: CommandPopup,
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
    pub selected_index: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Market {
    pub title: &'static str,
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
pub enum TopPageAction {
    SelectPane(u32),
}

pub fn top_page_reducer(top_page: &mut TopPage, action: &TopPageAction, _env: &Env) {
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
                selected_index: 0,
                markets: vec![
                    Market {
                        title: "Will BTC hit 100k in 2026?",
                        slug: "btc-100k-2026".into(),
                        bookmarked: true,
                        yes_market_price: 63.0,
                        no_market_price: 38.0,
                        volume24h: 842_000.0,
                        movement: 4.0,
                        spread: 2.0,
                    },
                    Market {
                        title: "Fed cuts rates by Sep?",
                        slug: "fed-cuts-sep".into(),
                        bookmarked: false,
                        yes_market_price: 41.0,
                        no_market_price: 60.0,
                        volume24h: 611_000.0,
                        movement: -2.0,
                        spread: 3.0,
                    },
                    Market {
                        title: "Lakers win tonight?",
                        slug: "lakers-win-tonight".into(),
                        bookmarked: false,
                        yes_market_price: 55.0,
                        no_market_price: 46.0,
                        volume24h: 570_000.0,
                        movement: 1.0,
                        spread: 2.0,
                    },
                    Market {
                        title: "ETH ETF inflows above $1B?",
                        slug: "eth-etf-inflows-1b".into(),
                        bookmarked: false,
                        yes_market_price: 72.0,
                        no_market_price: 29.0,
                        volume24h: 510_000.0,
                        movement: 6.0,
                        spread: 4.0,
                    },
                    Market {
                        title: "Trump wins popular vote?",
                        slug: "trump-popular-vote".into(),
                        bookmarked: true,
                        yes_market_price: 49.0,
                        no_market_price: 52.0,
                        volume24h: 421_000.0,
                        movement: -1.0,
                        spread: 2.0,
                    },
                    Market {
                        title: "CPI below forecast?",
                        slug: "cpi-below-forecast".into(),
                        bookmarked: false,
                        yes_market_price: 36.0,
                        no_market_price: 65.0,
                        volume24h: 390_000.0,
                        movement: -5.0,
                        spread: 5.0,
                    },
                    Market {
                        title: "SpaceX launch this week?",
                        slug: "spacex-launch-week".into(),
                        bookmarked: false,
                        yes_market_price: 83.0,
                        no_market_price: 18.0,
                        volume24h: 311_000.0,
                        movement: 8.0,
                        spread: 3.0,
                    },
                    Market {
                        title: "Oil closes above $90?",
                        slug: "oil-above-90".into(),
                        bookmarked: false,
                        yes_market_price: 22.0,
                        no_market_price: 79.0,
                        volume24h: 280_000.0,
                        movement: -3.0,
                        spread: 4.0,
                    },
                ],
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
        }
    }
}
