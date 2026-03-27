use reqwest::Client;

pub struct DataStore {
    client: Client,
}

impl DataStore {
    pub fn new(client: Client) -> Self {
        Self {
            client: client,
        }
    }

    pub async fn get_prices(&self) -> Result<Vec<crypto_market::SymbolPrice>, reqwest::Error> {
        let prices = self.client.get("https://api.binance.com/api/v3/ticker/price")
            .send()
            .await?
            .json::<Vec<crypto_market::SymbolPrice>>()
            .await?;

        Ok(prices)
    }
}

pub mod crypto_market {
    #[derive(serde::Deserialize)]
    #[derive(Debug)]
    pub struct SymbolPrice {
        #[serde(rename = "symbol")]
        pub name: String,
        pub price: f64,
    }

    #[derive(Debug)]
    pub enum PricesState {
        Initial,
        Loading,
        Loaded(Vec<SymbolPrice>),
    }

    #[derive(Debug)]
    pub struct PricesFeatureState {
        pub prices: PricesState,
        pub last_update_tick_sec: u32,
    }
}