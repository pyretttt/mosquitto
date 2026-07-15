use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Clear, Paragraph},
};

use crate::features::window_size::WindowSize;

static EMPTY: &str = "";

pub fn draw_window_size(frame: &mut Frame, state: &WindowSize) {
    let area = frame.area();
    frame.render_widget(Clear, area);
    frame.render_widget(Block::new().style(Style::default().bg(Color::Black)), area);

    let [_, content_area, _] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(5),
        Constraint::Fill(1),
    ])
    .areas(area);

    let current = format!(
        "{}x{}",
        state.current_size.width, state.current_size.height
    );
    let required = format!(
        "{}x{}",
        state.required_size.width, state.required_size.height
    );

    let lines = vec![
        Line::from(Span::styled(
            "Terminal too small",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(EMPTY),
        Line::from(vec![
            Span::styled("Current:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(current, Style::default().fg(Color::Red)),
        ]),
        Line::from(vec![
            Span::styled("Required: ", Style::default().fg(Color::DarkGray)),
            Span::styled(required, Style::default().fg(Color::Green)),
        ]),
        Line::from(Span::styled(
            "Resize your terminal to continue",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    frame.render_widget(
        Paragraph::new(lines).alignment(Alignment::Center),
        content_area,
    );
}
