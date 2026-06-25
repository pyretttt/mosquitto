use reqwest::Client;

pub const GAMMA_API_BASE: &str = "https://gamma-api.polymarket.com";
pub const CLOB_API_BASE: &str = "https://clob.polymarket.com";

#[derive(Debug, Clone)]
pub struct ApiClient {
    http: Client,
}

impl Default for ApiClient {
    fn default() -> Self {
        Self {
            http: Client::new(),
        }
    }
}

impl ApiClient {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn http(&self) -> &Client {
        &self.http
    }
}
