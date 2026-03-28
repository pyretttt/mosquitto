use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Rect},
    style::{Color, Stylize},
    widgets::{Block, BorderType, Paragraph, Widget},
};

use crate::data;
use crate::app::AppState;

impl Widget for &AppState {
    /// Renders the user interface widgets.
    ///
    // This is where you add new widgets.
    // See the following resources:
    // - https://docs.rs/ratatui/latest/ratatui/widgets/index.html
    // - https://github.com/ratatui/ratatui/tree/master/examples
    fn render(self, area: Rect, buf: &mut Buffer) {
        match self.prices_feature.prices {
            data::crypto::PricesState::Loading | data::crypto::PricesState::Initial => {
                let text = "Loading...";
                let paragraph = Paragraph::new(text)
                    .block(block)
                    .fg(Color::Cyan)
                    .bg(Color::Black)
                    .centered();
                    }
            data::crypto::PricesState::Loaded(
                symbol_prices
            ) => todo!(),
            data::crypto::PricesState::PriceLoadFailed => todo!(),
        }
    }
}
