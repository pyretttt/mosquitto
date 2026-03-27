pub mod app;
pub mod event;
pub mod ui;
pub mod data;
pub mod functional;

const TICK_REFRESH_RATE: f64 = 30.0;


#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let mut terminal = ratatui::init();
    let mut env = app::Env::new(
        data::DataStore::new(reqwest::Client::new()),
        event::EventHandler::new(TICK_REFRESH_RATE),
        app::Settings {
            price_refresh_interval_seconds: 10,
            tick_refresh_rate: TICK_REFRESH_RATE,
        },
    );
    let mut app_state = app::AppState::default();
    app::run(&mut app_state, &mut env, &mut terminal).await?;
    ratatui::restore();

    color_eyre::Result::Ok(())
}
