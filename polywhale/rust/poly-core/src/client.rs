use polymarket_client_sdk_v2::clob::{
    Client as ClobClient,
    Config,
};
use polymarket_client_sdk_v2::gamma::{Client as GammaClient};
use polymarket_client_sdk_v2::error::Error;
use polymarket_client_sdk_v2::gamma::types::request::{
    MarketsRequest
};

#[derive(Debug, Clone)]
pub struct PolymarketClient {
    clob_client: ClobClient,
    gamma_client: GammaClient,
}

#[derive(Debug)]
pub struct PolyError(pub Error);


impl Default for PolymarketClient {
    fn default() -> Self {
        Self {
            // By default crate uses unresolvable endpoint
            clob_client: ClobClient::new("https://clob.polymarket.com", Config::default()).expect("Failed to create ClobClient"),
            gamma_client: GammaClient::default(),
        }
    }
}

impl PolymarketClient {
    pub async fn markets(
        &self,
        limit: i32,
        offset: i32
    ) -> Result<Vec<polymarket_client_sdk_v2::gamma::types::response::Market>, PolyError> {
        self.gamma_client.markets(
            &MarketsRequest::builder()
            .offset(offset)
            .limit(limit)
            .ascending(false)
            .build()
        ).await
        .map_err(PolyError)
    }
}