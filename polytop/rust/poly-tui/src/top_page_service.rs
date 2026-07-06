use poly_core::client::{PolymarketClient, PolyError};

const MARKETS_PER_PAGE: i32 = 20;

pub trait TopPageSvc: Send + Sync + Clone {
    fn load_markets(
        &self,
        next_cursor: i32
    ) -> impl Future<Output = Result<MarketsData, PolyError>> + Send;
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
    async fn load_markets(
        &self,
        next_cursor: i32
    ) -> Result<MarketsData, PolyError> {
        let markets = self.client.markets(MARKETS_PER_PAGE, next_cursor).await?;
        let markets_data = markets.into_iter().filter_map(|market| {
            let prices = market.outcome_prices?.clone();
            let (yes_market_price, no_market_price) = if market.outcomes.as_ref()?.get(0)?.to_lowercase() == "yes" {
                (prices.get(0)?, prices.get(1)?)
            } else if market.outcomes.as_ref()?.get(0)?.to_lowercase() == "no" {
                (prices.get(1)?, prices.get(0)?)
            } else {
                log::error!(target: "app", "Unexpected outcome: {:?} for market {:?}", market.outcomes, market.slug);
                return None;
            };
            Some(Market {
                title: market.question?.clone(),
                slug: market.slug?.clone(),
                bookmarked: false,
                yes_market_price: yes_market_price.as_f64(),
                no_market_price: no_market_price.as_f64(),
                volume24h: market.volume_24hr?.as_f64(),
                movement24h: market.one_day_price_change?.as_f64(),
                spread: market.spread?.as_f64(),
            })
        });
        Ok(MarketsData {
            markets: markets_data.collect(),
            next_cursor: next_cursor + MARKETS_PER_PAGE,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct State {
    pub markets_data: MarketsData,
    pub selected_market: u32,
    pub selected_market_summary: SelectedMarket,
    pub chart_activity: ChartActivity,
}

#[derive(Clone, Debug, Default)]
pub struct MarketsData {
    pub markets: Vec<Market>,
    pub next_cursor: i32,
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