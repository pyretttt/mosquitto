use ratatui::prelude::Size;

#[derive(Clone, Debug, Default)]
pub struct WindowSize {
    pub current_size: Size,
    pub required_size: Size,
}

impl WindowSize {
    pub fn new(required_size: Size) -> Self {
        Self {
            current_size: Size::default(),
            required_size: required_size,
        }
    }
}