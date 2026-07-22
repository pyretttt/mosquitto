use poly_core::client::{PolymarketClient, PolyError};

const EVENTS_PER_PAGE: i32 = 30;

pub trait TopPageSvc: Send + Sync + Clone {
    fn load_events(
        &self,
        next_cursor: i32,
    ) -> impl Future<Output = Result<EventsData, PolyError>> + Send;
}

#[derive(Clone, Debug)]
pub struct TopPageService {
    pub client: PolymarketClient,
}

impl TopPageService {
    pub fn new(client: PolymarketClient) -> Self {
        Self {
            client,
        }
    }
}

impl TopPageSvc for TopPageService {
    async fn load_events(&self, next_cursor: i32) -> Result<EventsData, PolyError> {
        let events = self.client.events(EVENTS_PER_PAGE, next_cursor).await?;
        let events = events
            .into_iter()
            .map(|event| {
                let markets = event
                    .markets
                    .unwrap_or_default()
                    .into_iter()
                    .filter_map(|market| {
                        let prices = market.outcome_prices?.clone();
                        let (yes_market_price, no_market_price) =
                            if market.outcomes.as_ref()?.get(0)?.to_lowercase() == "yes" {
                                (prices.get(0)?, prices.get(1)?)
                            } else if market.outcomes.as_ref()?.get(0)?.to_lowercase() == "no" {
                                (prices.get(1)?, prices.get(0)?)
                            } else {
                                log::error!(
                                    target: "app",
                                    "Unexpected outcome: {:?} for market {:?}",
                                    market.outcomes,
                                    market.slug
                                );
                                return None;
                            };
                        let resolution_status: Option<UmaResolutionStatus> = match &market.uma_resolution_status {
                            Some(status) => status.try_into().map_or(None, |s| Some(s)),
                            None => None,
                        };
                        Some(Market::new(
                            market.question.unwrap_or_else(|| "N/A".to_owned()),
                            market.slug.unwrap_or_else(|| "N/A".to_owned()),
                            false,
                            yes_market_price.as_f64(),
                            no_market_price.as_f64(),
                            market.volume_24hr.unwrap_or_default().as_f64(),
                            market.one_day_price_change.unwrap_or_default().as_f64(),
                            market.spread?.as_f64(),
                            resolution_status
                        ))
                    })
                    .collect::<Vec<_>>();
                let markets_count = markets.len();
                let volume24h = event.volume_24hr.unwrap_or_default().as_f64();
                Event::new(
                    event.id,
                    event.title.unwrap_or_else(|| "N/A".to_owned()),
                    event.slug.unwrap_or_else(|| "N/A".to_owned()),
                    false,
                    volume24h,
                    markets,
                    markets_count,
                )
            })
            .collect::<Vec<_>>();
        Ok(EventsData {
            events,
            next_cursor: next_cursor + EVENTS_PER_PAGE,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct State {
    pub events_data: EventsData,
    pub selected_market: u32,
    pub selected_market_summary: SelectedMarket,
    pub chart_activity: ChartActivity,
}

#[derive(Clone, Debug, Default)]
pub struct EventsData {
    pub events: Vec<Event>,
    pub next_cursor: i32,
}

#[derive(Clone, Debug, Default)]
pub struct Event {
    pub id: String,
    pub title: String,
    pub slug: String,
    pub bookmarked: bool,
    pub volume24h: f64,
    pub markets: Vec<Market>,
    pub markets_count: usize,
    pub rank_label: String,
    pub bookmark_label: &'static str,
    pub volume_label: String,
    pub markets_count_label: String,
}

impl Event {
    pub fn new(
        id: String,
        title: String,
        slug: String,
        bookmarked: bool,
        volume24h: f64,
        markets: Vec<Market>,
        markets_count: usize,
    ) -> Self {
        Self {
            id,
            title,
            slug,
            bookmarked,
            volume24h,
            markets,
            markets_count,
            rank_label: String::default(),
            bookmark_label: if bookmarked { "★" } else { "" },
            volume_label: format_volume_compact(volume24h),
            markets_count_label: format!("{}mkt", markets_count),
        }
    }

    pub fn set_rank(&mut self, rank: usize) {
        self.rank_label = rank.to_string();
    }

    pub fn set_bookmarked(&mut self, bookmarked: bool) {
        self.bookmarked = bookmarked;
        self.bookmark_label = if bookmarked { "★" } else { "" };
    }
}

#[derive(Clone, Debug, Default)]
pub struct Market {
    pub title: String,
    pub slug: String,
    pub bookmarked: bool,
    pub yes_market_price: f64,
    pub no_market_price: f64,
    pub volume24h: f64,
    pub movement24h: f64,
    pub spread: f64,
    pub rank_label: String,
    pub bookmark_label: &'static str,
    pub yes_label: String,
    pub no_label: String,
    pub volume_label: String,
    pub movement_label: String,
    pub movement_kind: ActivityKind,
    pub spread_label: String,
    pub resolution_status: Option<UmaResolutionStatus>,
}

#[derive(Clone, Debug)]
pub enum UmaResolutionStatus {
    Pending,
    Resolved,
    Failed,
}

impl TryFrom<&String> for UmaResolutionStatus {
    type Error = String;
    fn try_from(value: &String) -> Result<Self, String> {
        match value.to_lowercase().as_str() {
            "pending" => Ok(UmaResolutionStatus::Pending),
            "resolved" => Ok(UmaResolutionStatus::Resolved),
            "failed" => Ok(UmaResolutionStatus::Failed),
            _ => Err("Failed to parse UmaResolutionStatus".to_owned()),
        }
    }
}

impl Market {
    pub fn new(
        title: String,
        slug: String,
        bookmarked: bool,
        yes_market_price: f64,
        no_market_price: f64,
        volume24h: f64,
        movement24h: f64,
        spread: f64,
        resolution_status: Option<UmaResolutionStatus>,
    ) -> Self {
        let movement_label = format_movement(movement24h);
        let movement_kind = movement_kind(movement24h);
        Self {
            title,
            slug,
            bookmarked,
            yes_market_price,
            no_market_price,
            volume24h,
            movement24h,
            spread,
            rank_label: String::default(),
            bookmark_label: if bookmarked { "★" } else { "" },
            yes_label: format_cents(yes_market_price),
            no_label: format_cents(no_market_price),
            volume_label: format_volume_compact(volume24h),
            movement_label,
            movement_kind,
            spread_label: format_cents(spread),
            resolution_status,
        }
    }

    pub fn set_rank(&mut self, rank: usize) {
        self.rank_label = rank.to_string();
    }

    pub fn set_bookmarked(&mut self, bookmarked: bool) {
        self.bookmarked = bookmarked;
        self.bookmark_label = if bookmarked { "★" } else { "" };
    }
}

#[derive(Clone, Debug, Default)]
pub struct SelectedMarket {
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
    pub yes_label: String,
    pub no_label: String,
    pub yes_quotes_label: String,
    pub no_quotes_label: String,
    pub volume_label: String,
    pub liquidity_label: String,
    pub open_interest_label: String,
}

impl SelectedMarket {
    pub fn new(
        slug: String,
        yes_market_price: f64,
        no_market_price: f64,
        yes_bid: f64,
        yes_ask: f64,
        no_bid: f64,
        no_ask: f64,
        spread: f64,
        volume24h: f64,
        liquidity: f64,
        open_interest: f64,
        end_date: String,
    ) -> Self {
        Self {
            slug,
            yes_market_price,
            no_market_price,
            yes_bid,
            yes_ask,
            no_bid,
            no_ask,
            spread,
            volume24h,
            liquidity,
            open_interest,
            end_date,
            yes_label: format_cents(yes_market_price),
            no_label: format_cents(no_market_price),
            yes_quotes_label: format!(
                "  bid {} / ask {}   spread {}",
                format_cents(yes_bid),
                format_cents(yes_ask),
                format_cents(spread),
            ),
            no_quotes_label: format!(
                "  bid {} / ask {}",
                format_cents(no_bid),
                format_cents(no_ask),
            ),
            volume_label: format_dollar_compact(volume24h),
            liquidity_label: format_dollar_compact(liquidity),
            open_interest_label: format_dollar_compact(open_interest),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChartActivity {
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

fn format_cents(value: f64) -> String {
    format!("{:.3}¢", value)
}

fn format_movement(value: f64) -> String {
    if value >= 0.0 {
        format!("+{:.3}¢", value)
    } else {
        format!("{:.3}¢", value)
    }
}

fn movement_kind(value: f64) -> ActivityKind {
    if value > 0.0 {
        ActivityKind::Positive
    } else if value < 0.0 {
        ActivityKind::Negative
    } else {
        ActivityKind::Muted
    }
}

fn format_volume_compact(value: f64) -> String {
    if value >= 1_000_000.0 {
        format!("{:.0}M", value / 1_000_000.0)
    } else if value >= 1_000.0 {
        format!("{:.0}k", value / 1_000.0)
    } else {
        format!("{:.0}", value)
    }
}

fn format_dollar_compact(value: f64) -> String {
    if value >= 1_000_000.0 {
        format!("${:.1}M", value / 1_000_000.0)
    } else if value >= 1_000.0 {
        format!("${:.1}k", value / 1_000.0)
    } else {
        format!("${:.0}", value)
    }
}
