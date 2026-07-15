use std::env;
use std::sync::OnceLock;

pub struct Config {
    pub tick_rate: f64,
}

impl Config {
    pub fn new() -> Self {
        Self {
            tick_rate: env::var("TICK_RATE")
                .unwrap_or("30".to_owned())
                .parse::<f64>()
                .expect("TICK_RATE must be a valid number"),
        }
    }
}

pub fn get_config() -> &'static Config {
    static CONFIG: OnceLock<Config> = OnceLock::new();
    CONFIG.get_or_init(|| Config::new())
}