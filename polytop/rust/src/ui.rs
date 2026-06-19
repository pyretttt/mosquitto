use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::models::app_state::AppState;

pub fn draw(frame: &mut Frame, state: &mut AppState) {
    let text_area = Rect::new(0, 0, frame.size().width, frame.size().height);
    let text = Span::styled(format!("counter: {}", state.counter), Style::default().fg(Color::White));

    frame.render_widget(
        text,
        text_area,
    );
}
