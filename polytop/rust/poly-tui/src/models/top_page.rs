
static TOP_PAGE_TITLE: &str = "Polytop";

#[derive(Clone, Debug, Default)]
pub struct TopPage {
    pub title: &'static str,
    pub text: String,
}

#[derive(Clone, Debug, Default)]
pub struct MarketHeat {
    title: &'static str,
    pub volume24h: f64,
    pub open_interest: f64,
    pub active_markets: u64,
    pub biggest_move: f64,
    pub categories: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct TopMarkets {
    pub title: &'static str,
    // pub markets: Vec<Market>,
}