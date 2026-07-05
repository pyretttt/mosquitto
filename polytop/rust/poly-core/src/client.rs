use polymarket_client_sdk_v2::clob::{
    Client as ClobClient,
    Config,
    types::response::{MarketResponse, Page}
};
use polymarket_client_sdk_v2::error::Error;

#[derive(Debug, Clone)]
pub struct PolymarketClient {
    clob_client: ClobClient,
}

#[derive(Debug)]
pub struct PolyError(pub Error);


impl Default for PolymarketClient {
    fn default() -> Self {
        Self {
            clob_client: ClobClient::new("https://clob.polymarket.com", Config::default()).expect("Failed to create ClobClient"),
        }
    }
}

impl PolymarketClient {
    pub async fn markets(&self, next_cursor: Option<String>) -> Result<Page<MarketResponse>, PolyError> {
        self.clob_client.markets(next_cursor).await
            .map_err(PolyError)
    }
}