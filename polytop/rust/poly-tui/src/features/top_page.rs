use std::error::Error;
use std::fmt;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::widgets::TableState;

use crate::pair::Pair;
use crate::features::app::Action;
use crate::event::Event;
use crate::env::Env;
pub use crate::top_page_service::Market;
use crate::top_page_service::{
    ActivityEntry, ActivityKind, ChartActivity,
    MarketsData, SelectedMarket, TopPageSvc,
};

static TOP_PAGE_TITLE: &str = " POLYTOP ";

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
    pub latency_label: String,
    pub refresh_label: String,
    pub mode_label: String,
}

impl StatusPane {
    pub fn refresh_labels(&mut self) {
        self.latency_label = format!("   latency {}ms", self.latency);
        self.refresh_label = format!("   refresh {}ms", self.refresh_interval);
        self.mode_label = format!("   mode {}", self.mode);
    }
}

#[derive(Clone, Debug, Default)]
pub struct MarketsPane {
    pub title: &'static str,
    pub filter: String,
    pub title_label: String,
    pub footer_label: String,
    pub markets_data: MarketsData,
    pub table_state: TableState,
}

impl MarketsPane {
    pub fn refresh_labels(&mut self) {
        self.title_label = format!(" [1] - {}: {} ", self.title, self.filter);
        self.footer_label = format!(
            " {} markets | j/k move | b bookmarks/all | w bookmark ",
            self.markets_data.markets.len(),
        );
    }
}

#[derive(Clone, Debug, Default)]
pub struct MarketSummary {
    pub title: &'static str,
    pub selected_market: SelectedMarket,
}

#[derive(Clone, Debug, Default)]
pub struct ChartActivityPane {
    pub title: &'static str,
    pub title_label: String,
    pub chart_activity: ChartActivity,
}

#[derive(Clone, Debug, Default)]
pub struct CommandPopup {
    pub filter: String,
    pub sort: String,
    pub mode: String,
    pub status_label: String,
}

impl CommandPopup {
    pub fn refresh_labels(&mut self) {
        self.status_label = format!(
            "Filter: {} sort:{}                               {}",
            self.filter, self.sort, self.mode,
        );
    }
}

#[derive(Clone, Debug)]
pub struct MarketLoadResult {
    pub markets_data: MarketsData,
    pub session: String,
}

#[derive(Clone, Debug)]
pub enum TopPageAction {
    SelectPane(u32),
    MarketsLoadRequested,
    MarketsRegularRequestFinished(Result<MarketLoadResult, TopPageError>),
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
            let top_page_svc = env.top_page_svc.clone();
            let markets_next_cursor = top_page.markets_pane.markets_data.next_cursor.clone();
            env.fire_and_forget(async move {
                match top_page_svc.load_markets(markets_next_cursor, &[]).await {
                    Ok(markets) => {
                        _ = sender.send(
                            TopPageAction::MarketsRegularRequestFinished(
                                Ok(MarketLoadResult { markets_data: markets, session: current_session } )
                            ).into()
                        );
                    },
                    Err(err) => {
                        log::error!(target: "app", "[TopPage] MarketsRequestFailed: {:?}", err);
                        _ = sender.send(TopPageAction::MarketsRegularRequestFinished(Err(TopPageError::MarketsRequestFailed)).into());
                    }
                }
            });
        },
        TopPageAction::MarketsRegularRequestFinished(result) => {
            match result {
                Ok(load_result) => {
                    if let Some(ref session) = top_page.markets_load_session && load_result.session.eq(session) {
                        let markets = &mut top_page.markets_pane.markets_data.markets;
                        let start_rank = markets.len() + 1;
                        for (offset, market) in load_result.markets_data.markets.iter_mut().enumerate() {
                            market.set_rank(start_rank + offset);
                        }
                        markets.append(&mut load_result.markets_data.markets);
                        top_page.markets_pane.markets_data.next_cursor = load_result.markets_data.next_cursor;
                        top_page.markets_pane.refresh_labels();
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
            top_page.markets_load_session = None;
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
                if x == 'j' && self.markets_pane.markets_data.markets.len() > selected_idx + 1 {
                    self.markets_pane.table_state.select_next();
                    let count = self.markets_pane.markets_data.markets.len();
                    if self.markets_pane.table_state.selected().map_or(false, |idx| idx >= count - 1) {
                        log::info!(target: "app", "TopPage: Loading more markets");
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
        let mut data = Self {
            left_title: TOP_PAGE_TITLE,
            right_title: String::new(),
            current_pane: 1,
            status_pane: StatusPane {
                is_online: true,
                ws_live: true,
                latency: 42,
                refresh_interval: 500,
                mode: "observe".into(),
                latency_label: String::new(),
                refresh_label: String::new(),
                mode_label: String::new(),
            },
            markets_pane: MarketsPane {
                title: "Top Markets",
                filter: "all".into(),
                title_label: String::new(),
                footer_label: String::new(),
                markets_data: MarketsData {
                    markets: vec![],
                    next_cursor: 0,
                },
                table_state: TableState::default().with_selected(Some(0)),
            },
            selected_market_pane: MarketSummary {
                title: "Will BTC hit 100k in 2026?",
                selected_market: SelectedMarket::new(
                    "btc-100k-2026".into(),
                    63.0,
                    38.0,
                    62.0,
                    64.0,
                    37.0,
                    39.0,
                    2.0,
                    842_100.0,
                    184_300.0,
                    2_400_000.0,
                    "2026-12-31".into(),
                ),
            },
            chart_activity_pane: ChartActivityPane {
                title: "Chart + Activity",
                title_label: " [3] - Chart + Activity ".into(),
                chart_activity: ChartActivity {
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
            },
            command_popup: CommandPopup {
                filter: "politics volume>100k".into(),
                sort: "move".into(),
                mode: "NORMAL".into(),
                status_label: String::new(),
            },
            error_msg: None,
            markets_load_session: None,
            is_loading: false,
        };

        data.markets_pane.markets_data.markets.reserve(50);
        data.markets_pane.refresh_labels();
        data.command_popup.refresh_labels();
        data.status_pane.refresh_labels();
        data
    }
}

impl fmt::Display for TopPageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{:?}", self)
    }
}

impl Error for TopPageError {}