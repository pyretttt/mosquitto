use polymarket_client_sdk_v2::clob::{
    Client as ClobClient,
    types::response::{MarketResponse, Page}
};
use polymarket_client_sdk_v2::error::Error;

#[derive(Debug, Clone)]
pub struct PolymarketClient {
    clob_client: ClobClient,
}

impl Default for PolymarketClient {
    fn default() -> Self {
        Self {
            clob_client: ClobClient::default(),
        }
    }
}

impl PolymarketClient {
    pub async fn markets(&self, next_cursor: Option<String>) -> Result<Page<MarketResponse>, Error> {
        self.clob_client.markets(next_cursor).await
    }
}