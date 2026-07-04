pub mod config;
pub mod env;
pub mod event;
pub mod features;
pub mod ui;
pub mod ui_components;
pub mod pair;

use env::Env;
use features::app::{AppState};
use ui::run;
use event::sidecar_event_loop;

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let mut terminal = ratatui::init();
    let mut env = Env::new();

    env.fire_and_forget(sidecar_event_loop(env.sender.clone()));

    run(
        &mut AppState::default(),
        &mut env,
        &mut terminal
    ).await?;
    ratatui::restore();
    Ok(())
}
