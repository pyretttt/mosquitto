use reqwest::Client;
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

    pub async fn get_price_change(
        &self,
        symbol: &str,
        window_size: &str,
    ) -> Result<crypto::PriceChange, reqwest::Error> {
        let price_chage = self.client.get(
            format!("https://api.binance.com/api/v3/ticker?symbol={}&windowSize={}&type=FULL", symbol, window_size)
        )
        .send()
        .await?
        .json::<crypto::PriceChange>()
        .await?;

        Ok(price_chage)
    }
}

pub mod crypto {
    #[derive(serde::Deserialize)]
    #[derive(Debug, Clone)]
    pub struct SymbolPrice {
        #[serde(rename = "symbol")]
        pub name: String,
        #[serde(deserialize_with = "serde_aux::field_attributes::deserialize_number_from_string")]
        pub price: f64,
    }

    #[derive(Debug, Clone)]
    pub enum PricesLoadingState {
        Loading,
        PriceLoadFailed,
        Idle,
    }

    #[derive(Debug, Clone)]
    pub struct PricesFeatureState {
        pub loading: PricesLoadingState,
        pub prices: Vec<SymbolPrice>,
        pub last_update_tick_sec: u32,
        pub selected_index: usize,
    }

    #[derive(Debug, Clone, serde::Deserialize)]
    pub struct PriceChange {
        pub symbol: String,
        #[serde(rename = "priceChange")]
        #[serde(deserialize_with = "serde_aux::field_attributes::deserialize_number_from_string")]
        pub price_change: f64,
        #[serde(rename = "priceChangePercent")]
        #[serde(deserialize_with = "serde_aux::field_attributes::deserialize_number_from_string")]
        pub price_change_percent: f64,
        #[serde(rename = "weightedAvgPrice")]
        #[serde(deserialize_with = "serde_aux::field_attributes::deserialize_number_from_string")]
        pub weighted_avg_price: f64,
        #[serde(rename = "openPrice")]
        #[serde(deserialize_with = "serde_aux::field_attributes::deserialize_number_from_string")]
        pub open_price: f64,
        #[serde(rename = "highPrice")]
        #[serde(deserialize_with = "serde_aux::field_attributes::deserialize_number_from_string")]
        pub high_price: f64,
        #[serde(rename = "lowPrice")]
        #[serde(deserialize_with = "serde_aux::field_attributes::deserialize_number_from_string")]
        pub low_price: f64,
        #[serde(rename = "lastPrice")]
        #[serde(deserialize_with = "serde_aux::field_attributes::deserialize_number_from_string")]
        pub last_price: f64,
        #[serde(rename = "volume")]
        #[serde(deserialize_with = "serde_aux::field_attributes::deserialize_number_from_string")]
        pub volume: f64,
        #[serde(rename = "quoteVolume")]
        #[serde(deserialize_with = "serde_aux::field_attributes::deserialize_number_from_string")]
        pub quote_volume: f64,
    }
}