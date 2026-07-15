use ratatui::{
    Frame,
    style::{Color, Style},
    layout::{Layout, Constraint},
    text::{Text},
    widgets::{Widget, Clear, Block},
};
use tui_logger::*;

use crate::features::log_page::{LogPage};

pub fn draw_log_page(frame: &mut Frame, log_page: &LogPage) {
    frame.render_widget(Clear, frame.area());
    frame.render_widget(Block::new().style(Style::default().bg(Color::Black)), frame.area());


    let [log_area, help_area] = Layout::vertical(
        [Constraint::Fill(1), Constraint::Length(4)]
    ).areas(frame.area());

    TuiLoggerSmartWidget::default()
        .style_error(Style::default().fg(Color::Red))
        .style_debug(Style::default().fg(Color::Green))
        .style_warn(Style::default().fg(Color::Yellow))
        .style_trace(Style::default().fg(Color::Magenta))
        .style_info(Style::default().fg(Color::Cyan))
        .output_separator(':')
        .output_timestamp(Some("%H:%M:%S".to_string()))
        .output_level(Some(TuiLoggerLevelOutput::Abbreviated))
        .output_target(true)
        .output_file(true)
        .output_line(true)
        .state(&log_page.logs_state.0)
        .render(log_area, frame.buffer_mut());

    frame.render_widget(
        Text::from(log_page.help.as_slice())
        .style(Color::Gray)
        .centered(),
        help_area
    );
}
