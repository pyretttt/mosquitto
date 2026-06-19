use tokio::sync::mpsc;


pub struct AppState {
    pub page: Page,
}

pub struct Env {
    pub action_tx: mpsc::UnboundedSender<Action>,
    pub action_rx: mpsc::UnboundedReceiver<Action>,
}

impl Env {
    pub fn new() -> Self {
        let (action_tx, action_rx) = mpsc::unbounded_channel::<Action>();
        Self {
            action_tx,
            action_rx,
        }
    }
}

pub fn app_state_reduce(app_state: AppState, action: Action, env: Env) -> AppState {
    let mut new_state = AppState {
        ..app_state
    };

    match action {
        Action::NextPage => {},
    }

    return new_state;
}

pub enum Action {
    NextPage,
}

pub enum Page {
    Intro(IntroPage),
}

pub struct IntroPage {
    pub title: String,
    pub text: String,
}