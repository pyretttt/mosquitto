use reqwest::Client;
use std::collections::HashMap;

pub struct DataStore {
    client: Client,
}

pub mod crypto_market {
    #[derive(serde::Deserialize)]
    #[derive(Debug)]
    pub struct SymbolPrice {
        #[serde(rename = "symbol")]
        pub name: String,
        pub price: f64,
    }
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