use ratatui::{
    Frame, 
    style::{Color, Style}, 
    widgets::{Block, Widget},
};
use tui_logger::*;

use crate::features::log_page::{LogPage};

pub fn draw_log_page(frame: &mut Frame, log_page: &LogPage) {
    TuiLoggerWidget::default()
        .block(Block::bordered().title("Logs"))
        .output_separator('|')
        .output_timestamp(Some("%F %H:%M:%S%.3f".to_string()))
        .output_level(Some(TuiLoggerLevelOutput::Long))
        .output_target(false)
        .output_file(false)
        .output_line(false)
        .style(Style::default().fg(Color::White))
        .state(&log_page.state.0)
        .render(frame.area(), frame.buffer_mut());
}
