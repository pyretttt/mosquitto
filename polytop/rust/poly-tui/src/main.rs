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

use tui_logger;

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    setup_logging();
    
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

fn setup_logging() {
    tui_logger::init_logger(log::LevelFilter::Trace).unwrap();
    tui_logger::set_default_level(log::LevelFilter::Trace);

    let mut dir = std::env::temp_dir();
    dir.push("tui-logger_demo.log");
    let file_options = tui_logger::TuiLoggerFile::new(dir.to_str().unwrap())
        .output_level(Some(tui_logger::TuiLoggerLevelOutput::Abbreviated))
        .output_file(true)
        .output_separator(':');
    tui_logger::set_log_file(file_options);
}