use std::error::Error;
use std::fmt;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::widgets::TableState;
use ratatui::prelude::Size;

use crate::pair::Pair;
use crate::features::app::Action;
use crate::event::Event;
use crate::env::Env;
pub use crate::top_page_service::{Event as PolyEvent, Market};
use crate::top_page_service::{
    ActivityEntry, ActivityKind, ChartActivity, EventsData, SelectedMarket, TopPageSvc,
};

static TOP_PAGE_TITLE: &str = " POLYWHALE ";

/// Max visible rows in the markets table (scrolls when there are more).
pub const MAX_MARKETS_TABLE_HEIGHT: usize = 6;

#[derive(Clone, Debug)]
pub struct TopPage {
    pub left_title: &'static str,
    pub right_title: String,

    pub error_msg: Option<Pair<String, String>>,
    pub current_pane: u32,

    pub status_pane: StatusPane,
    pub events_pane: EventsPane,
    pub selected_market_pane: MarketSummary,
    pub chart_activity_pane: ChartActivityPane,
    pub command_popup: CommandPopup,
    pub events_load_session: Option<String>,
    pub is_loading: bool,
    pub ui_window_size: Size,
}

impl TopPage {
    pub fn update_window_size(&mut self, new_size: Size) {
        self.ui_window_size = new_size;
        self.events_pane.window_size = SizeExt(new_size).get_events_window_size();
    }

    pub fn table_payload_height(&self) -> u16 {
        let events_h = self
            .events_pane
            .event_slice()
            .len()
            .min(self.events_pane.window_size) as u16;
        // Reserve space for the markets table so it stays visible (incl. when the
        // last event in the window is selected).
        (events_h + self.events_pane.markets_table_height()).max(1)
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
pub struct EventsPane {
    pub title: &'static str,
    pub filter: Option<Filter>,
    pub title_label: String,
    pub footer_label: String,
    pub events_data: EventsData,
    pub table_state: TableState,
    pub markets_table_state: TableState,
    pub markets_focused: bool,

    pub window_size: usize,
    pub offset: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Filter {
    pub search_term: String,
}

impl EventsPane {
    pub fn refresh_labels(&mut self) {
        self.title_label = format!(" [1] - {}: {} ", self.title, self.filter.as_ref().map_or("all", |f| &f.search_term));
        self.footer_label = if self.markets_focused {
            format!(
                " {} events | j/k markets | ←/esc events | enter focus ",
                self.events_data.events.len(),
            )
        } else {
            format!(
                " {} events | j/k events | →/enter markets | b bookmarks ",
                self.events_data.events.len(),
            )
        };
    }

    pub fn selected_event_idx(&self) -> Option<usize> {
        let abs = self.table_state.selected().unwrap_or(0) + self.offset;
        if abs < self.events_data.events.len() {
            Some(abs)
        } else {
            None
        }
    }

    pub fn selected_event(&self) -> Option<&PolyEvent> {
        self.selected_event_idx()
            .and_then(|idx| self.events_data.events.get(idx))
    }

    pub fn event_slice(&self) -> &[PolyEvent] {
        if self.offset >= self.events_data.events.len() {
            return &[];
        }
        let end = self.offset.saturating_add(self.window_size).min(self.events_data.events.len());
        &self.events_data.events[self.offset..end]
    }

    /// Visible height of the markets table (0..=MAX_MARKETS_TABLE_HEIGHT).
    pub fn markets_table_height(&self) -> u16 {
        self.selected_event()
            .map(|event| event.markets.len().min(MAX_MARKETS_TABLE_HEIGHT) as u16)
            .unwrap_or(0)
    }

    pub fn clear_market_focus(&mut self) {
        self.markets_focused = false;
        self.markets_table_state.select(None);
        self.refresh_labels();
    }

    pub fn enter_market_focus(&mut self) -> bool {
        if self.markets_focused { return true; }
        let Some(event) = self.selected_event() else {
            assert!(false, "Impossible branch");
            return false;
        };
        if event.markets.is_empty() {
            return false;
        }
        self.markets_focused = true;
        self.markets_table_state.select(Some(0));
        self.refresh_labels();
        true
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
pub struct EventLoadResult {
    pub events_data: EventsData,
    pub session: String,
}

#[derive(Clone, Debug)]
pub enum TopPageAction {
    SelectPane(u32),
    EventsLoadRequested,
    EventsRegularRequestFinished(Result<EventLoadResult, TopPageError>),
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
    EventsRequestFailed,
}

pub fn top_page_reducer(top_page: &mut TopPage, action: &mut TopPageAction, env: &Env) {
    match action {
        TopPageAction::SelectPane(pane) => {
            top_page.current_pane = *pane;
        },
        TopPageAction::EventsLoadRequested => {
            if top_page.events_load_session.is_some() { return; }
            let this_session = (env.gen_token)();
            top_page.is_loading = true;
            top_page.events_load_session = Some(this_session.clone());
            let sender = env.sender.clone();
            let top_page_svc = env.top_page_svc.clone();
            let events_next_cursor = top_page.events_pane.events_data.next_cursor;
            env.fire_and_forget(async move {
                match top_page_svc.load_events(events_next_cursor).await {
                    Ok(events_data) => {
                        _ = sender.send(
                            TopPageAction::EventsRegularRequestFinished(
                                Ok(EventLoadResult { events_data, session: this_session } )
                            ).into()
                        );
                    },
                    Err(err) => {
                        log::error!(target: "app", "[TopPage] EventsRequestFailed: {:?}", err);
                        _ = sender.send(TopPageAction::EventsRegularRequestFinished(Err(TopPageError::EventsRequestFailed)).into());
                    }
                }
            });
        },
        TopPageAction::EventsRegularRequestFinished(result) => {
            match result {
                Ok(load_result) => {
                    if let Some(ref session) = top_page.events_load_session && load_result.session.eq(session) {
                        let events = &mut top_page.events_pane.events_data.events;
                        let start_rank = events.len() + 1;
                        let was_empty = events.is_empty();
                        for (offset, event) in load_result.events_data.events.iter_mut().enumerate() {
                            event.set_rank(start_rank + offset);
                        }
                        events.append(&mut load_result.events_data.events);
                        top_page.events_pane.events_data.next_cursor = load_result.events_data.next_cursor;
                        top_page.events_pane.refresh_labels();
                        if was_empty && !top_page.events_pane.events_data.events.is_empty() {
                            top_page.events_pane.table_state.select(Some(0));
                        }
                    }
                },
                Err(_) => {
                    let current_token = (env.gen_token)();
                    let sender = env.sender.clone();
                    top_page.error_msg = Some(Pair::new(
                        "Failed to load events data, press `R` to retry".to_owned(),
                        current_token.clone()
                    ));
                    let sleep_fn = env.sleep.clone();
                    _ = env.fire_and_forget(async move {
                        sleep_fn.sleep(3000).await;
                        _ = sender.send(TopPageAction::HideErrorMsg { token: current_token }.into());
                    });
                }
            }
            top_page.events_load_session = None;
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
    fn move_event_selection(&mut self, down: bool, env: &Env) {
        let total = self.events_pane.events_data.events.len();
        let selected_in_window = self.events_pane.table_state.selected().unwrap_or(0);
        let abs = selected_in_window + self.events_pane.offset;

        if down {
            if abs >= total.saturating_sub(1) {
                if self.is_loading {
                    return;
                }
                log::info!(target: "app", "TopPage: Loading more events");
                _ = env.sender.send(TopPageAction::EventsLoadRequested.into());
                return;
            }
            if selected_in_window < self.events_pane.window_size - 1 {
                self.events_pane.table_state.select_next();
            } else {
                self.events_pane.offset = self.events_pane.offset.saturating_add(1);
            }
        } else {
            if abs == 0 {
                return;
            }
            if selected_in_window != 0 {
                self.events_pane.table_state.select_previous();
            } else {
                self.events_pane.offset = self.events_pane.offset.saturating_sub(1);
            }
        }
        self.events_pane.clear_market_focus();
        self.events_pane.markets_table_state = TableState::default();
    }

    pub fn key_input_middleware(&mut self, key_event: &KeyEvent, env: &Env) -> bool {
        match key_event.code {
            KeyCode::Char(x) if ['1', '2', '3'].contains(&x) => {
                self.current_pane = x.to_digit(10).unwrap_or(1);
                true
            },
            KeyCode::Enter | KeyCode::Right => {
                self.events_pane.enter_market_focus();
                true
            },
            KeyCode::Left | KeyCode::Esc => {
                if self.events_pane.markets_focused {
                    self.events_pane.clear_market_focus();
                    true
                } else {
                    false
                }
            },
            KeyCode::Char(x) if ['j', 'k'].contains(&x) => {
                if self.events_pane.events_data.events.len() == 0 { return false; }

                match self.events_pane.markets_focused {
                    true => {
                        let market_count = self
                            .events_pane
                            .selected_event()
                            .map(|e| e.markets.len())
                            .unwrap_or(0);
                        assert!(market_count != 0, "Impossible branch");
                        let market_idx = self
                            .events_pane
                            .markets_table_state
                            .selected()
                            .unwrap_or(0);

                        match x {
                            'j' => {
                                if market_idx >= market_count - 1 {
                                    // Edge: leave markets and move to next event.
                                    self.events_pane.clear_market_focus();
                                    self.move_event_selection(true, env);
                                } else {
                                    self.events_pane.markets_table_state.select(Some(market_idx + 1));
                                }
                            },
                            'k' => {
                                if market_idx == 0 {
                                    self.events_pane.clear_market_focus();
                                    self.move_event_selection(false, env);
                                } else {
                                    self.events_pane.markets_table_state.select(Some(market_idx - 1));
                                }
                            }
                            _ => assert!(false, "Impossible branch"),
                        }
                    },
                    false => {
                        match x {
                            'j' => {
                                self.move_event_selection(true, env);
                            },
                            'k' => {
                                self.move_event_selection(false, env);
                            },
                            _ => assert!(false, "Impossible branch"),
                        }
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
            events_pane: EventsPane {
                title: "Top Events",
                filter: Some(Filter {
                    search_term: "".into(),
                }),
                title_label: String::new(),
                footer_label: String::new(),
                events_data: EventsData {
                    events: vec![],
                    next_cursor: 0,
                },
                table_state: TableState::default().with_selected(Some(0)),
                markets_table_state: TableState::default(),
                markets_focused: false,
                offset: 0,
                window_size: SizeExt(ui_window_size).get_events_window_size(),
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
            events_load_session: None,
            is_loading: false,
        };

        data.events_pane.events_data.events.reserve(50);
        data.events_pane.refresh_labels();
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
    pub fn get_events_window_size(&self) -> usize {
        let window_height = self.0.height;
        log::info!(target: "app", "TopPage: window_height: {:?}", window_height);
        (window_height as f32 * 0.4).max(10.0).min(45.0) as usize
    }
}