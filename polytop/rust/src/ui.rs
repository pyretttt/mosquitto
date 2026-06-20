use ratatui::{
    Frame,
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};
use ratatui::DefaultTerminal;

use crate::models::app_state::{AppState, Page, app_reducer};
use crate::env::Env;

fn draw(frame: &mut Frame, state: &AppState) {
    let Page::Intro(intro) = &state.page;

    let [title_area, text_area, counter_area, help_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Fill(1),
    ])
    .margin(1)
    .areas(frame.area());

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            intro.title.as_str(),
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ))),
        title_area,
    );

    frame.render_widget(Paragraph::new(intro.text.as_str()), text_area);

    frame.render_widget(
        Paragraph::new(format!("counter: {}", state.counter)),
        counter_area,
    );

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Press q to quit",
            Style::default().fg(Color::DarkGray),
        ))),
        help_area,
    );
}

pub async fn run(
    app_state: &mut AppState,
    env: &mut Env,
    terminal: &mut DefaultTerminal,
) -> color_eyre::Result<()> {
    while app_state.running {
        terminal.draw(|frame| draw(frame, app_state))?;
        let mut event = env.receiver
            .recv()
            .await
            .ok_or(color_eyre::eyre::eyre!("Event receiver closed"))?;
        app_reducer(app_state, &mut event, env);
    }
    Ok(())
}
