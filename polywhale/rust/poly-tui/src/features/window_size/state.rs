use ratatui::prelude::Size;

#[derive(Clone, Debug, Default)]
pub struct WindowSize {
    pub current_size: Size,
    pub required_size: Size,

    pub labels: Labels,
}

impl WindowSize {
    pub fn update_labels(&mut self, current_size: Size) {
        self.current_size = current_size;
        self.labels.current_size = format!("{}x{}", current_size.width, current_size.height);
        self.labels.required_size = format!("{}x{}", self.required_size.width, self.required_size.height);
    }
}

#[derive(Clone, Debug, Default)]
pub struct Labels {
    pub current_size: String,
    pub required_size: String,
    pub title: &'static str,
    pub current: &'static str,
    pub required: &'static str,
    pub resize: &'static str,
}

impl WindowSize {
    pub fn new(required_size: Size) -> Self {
        Self {
            current_size: Size::default(),
            required_size: required_size,
            labels: Labels {
                title: "Terminal too small",
                current: "Current: ",
                required: "Required: ",
                resize: "Resize your terminal to continue",
                current_size: "".to_string(),
                required_size: "".to_string(),
            },
        }
    }
}