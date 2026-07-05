use poly_core::client::{PolymarketClient, PolyError};

pub trait TopController {
    
}

#[derive(Clone)]
pub struct TopControllerImpl {
    pub client: PolymarketClient,
}

impl TopControllerImpl {
    pub async fn load_markets(&mut self, _next_cursor: Option<String>) -> Result<Vec<Market>, PolyError> {
        Ok(vec![])
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
    pub movement: f64,
    pub spread: f64,
}
