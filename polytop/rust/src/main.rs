pub mod models;
pub mod event_loop;
pub mod config;
pub mod ui;

use std::io::{self, stdout};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph},
};
use tokio::time::{sleep, Duration};

use models::app_state::{AppState, Page, IntroPage, Env, app_state_reduce};
use event_loop::{EventLoop};


#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let result = run_app(&mut terminal).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

async fn run_app(terminal: &mut Terminal<CrosstermBackend<impl io::Write>>) -> color_eyre::Result<()> {
    let mut app_state = AppState {
        page: Page::Intro(IntroPage{title: "Polytop - Polymarket Monitor".to_owned(), text: "Hello, world!".to_owned()}),
        counter: 0,
    };
    let env = Env::new();
    tokio::spawn(async { env.event_loop.run().await });

    loop {
        terminal.draw(|frame| {
            ui::draw(frame, &mut app_state);
        })?;
        let event = env.event_loop.next().await?;
        match event {
            event_loop::Event::App(action) => {
                app_state_reduce(&mut app_state, action, &env);
            }
            _ => {
                continue;
            }
        }
    }

    Ok(())
}
