use reqwest::Client;
use serde;

pub struct Store {
    client: Client,
}

impl Store {
    pub fn new(client: Client) -> Self {
        Self {
            client: client,
        }
    }

    pub async fn get_prices(&self) -> Result<Vec<crypto::SymbolPrice>, reqwest::Error> {
        let prices = self.client.get("https://api.binance.com/api/v3/ticker/price")
            .send()
            .await?
            .json::<Vec<crypto::SymbolPrice>>()
            .await?;

        Ok(prices)
    }
}

pub mod crypto {
    #[derive(serde::Deserialize)]
    #[derive(Debug, Clone)]
    pub struct SymbolPrice {
        #[serde(rename = "symbol")]
        pub name: String,
        pub price: f64,
    }

    #[derive(Debug, Clone)]
    pub enum PricesState {
        Initial,
        Loading,
        Loaded(Vec<SymbolPrice>),
        PriceLoadFailed,
    }

    #[derive(Debug, Clone)]
    pub struct PricesFeatureState {
        pub prices: PricesState,
        pub last_update_tick_sec: u32,
    }
}