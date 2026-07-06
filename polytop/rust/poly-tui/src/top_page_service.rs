use poly_core::client::{PolymarketClient, PolyError};

pub trait TopPageSvc: Send + Sync + Clone {
    fn load_markets(
        &self,
        next_cursor: Option<String>
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
        next_cursor: Option<String>
    ) -> Result<MarketsData, PolyError> {
        let markets = self.client.markets(next_cursor).await?;
        let markets_data =markets.data.into_iter().map(|market| {
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
        Ok(MarketsData {
            markets: markets_data.collect(),
            next_cursor: Some(next_cursor),
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
    pub next_cursor: Option<String>,
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