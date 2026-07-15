pub mod config;
pub mod env;
pub mod event;
pub mod features;
pub mod ui;
pub mod ui_components;
pub mod pair;
pub mod top_page_service;

use env::Env;
use features::app::{AppState};
use ui::run;
use event::sidecar_event_loop;

use tui_logger;

const LOG_FILE_MAX_SIZE_BYTES: u64 = 1024 * 1024;

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    setup_logging();

    color_eyre::install()?;
    let mut terminal = Box::leak(Box::new(ratatui::init()));
    let mut env = Env::new(terminal.size().unwrap_or_default());

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
    dir.push("polywhale.log");

    // Simple log rotation by file size
    if std::fs::metadata(&dir).map(|meta| meta.len()).unwrap_or(0) >= LOG_FILE_MAX_SIZE_BYTES {
        std::fs::remove_file(&dir).unwrap();
    }

    let file_options = tui_logger::TuiLoggerFile::new(dir.to_str().unwrap())
        .output_level(Some(tui_logger::TuiLoggerLevelOutput::Abbreviated))
        .output_file(true)
        .output_separator(':');
    tui_logger::set_log_file(file_options);

    log::info!(target: "app", "Logging to: {:?}", dir);
}