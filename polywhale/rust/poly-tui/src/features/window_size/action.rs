use ratatui::prelude::Size;

#[derive(Clone, Debug)]
pub enum WindowSizeAction {
    Resize(Size),
}
