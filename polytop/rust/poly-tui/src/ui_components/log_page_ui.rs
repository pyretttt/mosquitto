use ratatui::{
    Frame, 
    style::{Color, Style}, 
    widgets::{Widget},
};
use tui_logger::*;

use crate::features::log_page::{LogPage};

pub fn draw_log_page(frame: &mut Frame, log_page: &LogPage) {
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
        .state(&log_page.state.0)
        .render(frame.area(), frame.buffer_mut());
}
