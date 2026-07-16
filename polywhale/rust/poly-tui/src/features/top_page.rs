use std::error::Error;
use std::fmt;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::widgets::TableState;
use ratatui::prelude::Size;

use crate::pair::Pair;
use crate::features::app::Action;
use crate::event::Event;
use crate::env::Env;
pub use crate::top_page_service::Market;
use crate::top_page_service::{
    ActivityEntry, ActivityKind, ChartActivity,
    MarketsData, SelectedMarket, TopPageSvc,
};

static TOP_PAGE_TITLE: &str = " POLYWHALE ";

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
    pub ui_window_size: Size,
}

impl TopPage {
    pub fn update_window_size(&mut self, new_size: Size) {
        self.ui_window_size = new_size;
        self.markets_pane.window_size = SizeExt(new_size).get_top_page_window_size();
    }
}

impl Default for TopPage {
    fn default() -> Self {
        Self::mock_data(Size::default())
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
    pub filter: Option<Filter>,
    pub title_label: String,
    pub footer_label: String,
    pub markets_data: MarketsData,
    pub table_state: TableState,

    pub window_size: usize,
    offset: usize
}

#[derive(Clone, Debug, Default)]
pub struct Filter {
    pub search_term: String,
}

impl MarketsPane {
    pub fn refresh_labels(&mut self) {
        self.title_label = format!(" [1] - {}: {} ", self.title, self.filter.as_ref().map_or("all", |f| &f.search_term));
        self.footer_label = format!(
            " {} markets | j/k move | b bookmarks/all | w bookmark ",
            self.markets_data.markets.len(),
        );
    }

    pub fn market_slice(&self) -> &[Market] {
        let offset = self.offset;
        let end = offset.saturating_add(self.window_size).min(self.markets_data.markets.len());
        &self.markets_data.markets[offset..end]
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
    Resize(Size),
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
        TopPageAction::Resize(size) => {
            top_page.update_window_size(*size);
        }
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
                let total_markets_count = self.markets_pane.markets_data.markets.len();
                if total_markets_count == 0 { return true; }
                let selected_idx_in_window = self.markets_pane.table_state.selected().unwrap_or(0);

                if x == 'j' {
                    if selected_idx_in_window.saturating_add(self.markets_pane.offset) == total_markets_count - 1 {
                        if self.is_loading { return true; }
                        log::info!(target: "app", "TopPage: Loading more markets");
                        _ = env.sender.send(TopPageAction::MarketsLoadRequested.into());
                    } else {
                        if selected_idx_in_window != self.markets_pane.window_size - 1 {
                            self.markets_pane.table_state.select_next();
                        } else {
                            self.markets_pane.offset = self.markets_pane.offset.saturating_add(1);
                        }
                    }
                } else if x == 'k' {
                    if selected_idx_in_window != 0 {
                        self.markets_pane.table_state.select_previous();
                    } else {
                        self.markets_pane.offset = self.markets_pane.offset.saturating_sub(1);
                    }
                }
                true
            },
            KeyCode::Char('f') => {
                true
            },
            _ => false,
        }
    }
}

// ================================================
// === Mock data
// ================================================
impl TopPage {
    pub fn mock_data(ui_window_size: Size) -> Self {
        let mut data = Self {
            ui_window_size: ui_window_size,
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
                filter: Some(Filter {
                    search_term: "".into(),
                }),
                title_label: String::new(),
                footer_label: String::new(),
                markets_data: MarketsData {
                    markets: vec![],
                    next_cursor: 0,
                },
                table_state: TableState::default().with_selected(Some(0)),
                offset: 0,
                window_size: SizeExt(ui_window_size).get_top_page_window_size(),
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

struct SizeExt(Size);

impl SizeExt {
    pub fn get_top_page_window_size(&self) -> usize {
        let window_height = self.0.height;
        log::info!(target: "app", "TopPage: window_height: {:?}", window_height);
        (window_height as f32 * 0.4).max(10.0).min(35.0) as usize
    }
}