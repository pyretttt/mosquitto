use polymarket_client_sdk_v2::clob::{
    Client as ClobClient,
    Config,
};
use polymarket_client_sdk_v2::gamma::{Client as GammaClient};
use polymarket_client_sdk_v2::error::Error;
use polymarket_client_sdk_v2::gamma::types::request::{
    EventsRequest, MarketsRequest, EventByIdRequest,
};
use polymarket_client_sdk_v2::gamma::types::response::{Event, Market};

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
    ) -> Result<Vec<Market>, PolyError> {
        self.gamma_client.markets(
            &MarketsRequest::builder()
            .offset(offset)
            .limit(limit)
            .ascending(false)
            .build()
        ).await
        .map_err(PolyError)
    }

    pub async fn events(
        &self,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<Event>, PolyError> {
        self.gamma_client.events(
            &EventsRequest::builder()
                .active(true)
                .closed(false)
                .offset(offset)
                .limit(limit)
                .ascending(false)
                .build(),
        )
        .await
        .map_err(PolyError)
    }

    pub async fn event_by_id(&self, id: &str) -> Result<Event, PolyError> {
        self.gamma_client
            .event_by_id(&EventByIdRequest::builder().id(id).build())
            .await
            .map_err(PolyError)
    }
}
