pub struct AppState {
    pub page: Page,
}

pub enum Page {
    Intro(IntroPage),
}

pub struct IntroPage {
    pub title: String,
    pub text: String,
}