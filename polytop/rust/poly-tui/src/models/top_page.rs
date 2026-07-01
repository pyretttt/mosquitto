
static TOP_PAGE_TITLE: &str = "Polytop";

#[derive(Clone, Debug, Default)]
pub struct TopPage {
    pub left_title: &'static str,
    pub right_title: String,

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