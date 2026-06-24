use std::time::Duration;
use std::assert;

use crossterm::event::{KeyCode, KeyEventKind, KeyModifiers};

use crate::env::Env;
use crate::event::{Event};
use crate::config::get_config;
use crate::models::command::{CommandPallette, Command};

#[derive(Clone, Debug)]
pub struct AppState {
    pub page: Page,
    pub running: bool,
    pub increment_token: Option<String>,
    pub command_pallette: Option<CommandPallette>,
}

#[derive(Clone, Debug)]
pub enum Page {
    Intro(IntroPage),
    LoadingPage(LoadingPage),
    Main(MainPage),
    Help(HelpPage),
}

#[derive(Clone, Debug)]
pub struct IntroPage {
}

#[derive(Clone, Debug, Default)]
pub struct MainPage {
    pub title: String,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct HelpPage {

}

const LOADING_TIPS: [&str; 4] = [
    "Press `q` to quit",
    "Press `/` to open command palette",
    "Press `?` to open help",
    "Press `Ctrl+c` to quit",
];

#[derive(Clone, Debug)]
pub struct LoadingPage {
    pub progress: f32,
    pub throbbler_state: throbber_widgets_tui::ThrobberState,
    pub loading_tip: &'static str,
    pub throbbler_caption: String,
    pub logo_color_index: usize,
    loading_tip_index: usize,
    tick: u16,
    pub is_finished: bool,
}

impl LoadingPage {
    pub fn new(throbbler_caption: impl Into<String>) -> Self {
        Self {
            progress: 0.0,
            loading_tip: LOADING_TIPS[0],
            throbbler_state: throbber_widgets_tui::ThrobberState::default(),
            throbbler_caption: throbbler_caption.into(),
            logo_color_index: 0,
            loading_tip_index: 0,
            tick: 0,
            is_finished: false,
        }
    }

    pub fn tick(&mut self, env: &Env) {
        self.tick = self.tick.wrapping_add(1);
        if self.tick % 3 == 0 {
            self.throbbler_state.calc_next();
            self.logo_color_index = self.logo_color_index.wrapping_add(1);
        }
        if self.tick % (get_config().tick_rate * 2.0) as u16 == 0 {
            self.loading_tip_index = (self.loading_tip_index + 1) % LOADING_TIPS.len();
            self.loading_tip = LOADING_TIPS[self.loading_tip_index];
        }
        if !self.is_finished && self.tick % 10 == 0 {
            let rng = (env.rng)(Some(0.01..0.05));
            self.progress = (self.progress + rng).min(0.87);
        }
    }
}

impl Default for LoadingPage {
    fn default() -> Self {
        Self::new("Cooking smooth performance...")
    }
}

#[derive(Clone, Debug)]
pub enum LoadingPageAction {
    Finished,
}

#[derive(Clone, Debug)]
pub enum Action {
    Next(String),
    CommandSent(Command),
    LoadingPage(LoadingPageAction),
    OpenPage(Page),
    Quit,
}

impl Into<Event> for Action {
    fn into(self) -> Event {
        Event::App(self)
    }
}

pub fn app_state_reduce(app_state: &mut AppState, action: &Action, env: &Env) {
    match action {
        Action::Next(ref token) if app_state.increment_token.as_ref() == Some(token) => {
            //
        },
        Action::CommandSent(command) => {
            match command {
                Command::Help => {
                    app_state.page = Page::Help(HelpPage {});
                },
                Command::Quit => app_state.running = false,
                Command::Intro => {
                    app_state.page = Page::Intro(IntroPage { });
                },
            }
        },
        Action::Next(_) => (),
        Action::Quit => app_state.running = false,
        Action::LoadingPage(action) => {
            if let Page::LoadingPage(ref mut loading) = app_state.page {
                loading_page_reducer(loading, action, env);
            } else {
                assert!(false, "Action dispatched to non-loading page");
            }
        },
        Action::OpenPage(page) => {
            app_state.page = page.clone();
        }
    }
}

pub fn app_reducer(app_state: &mut AppState, event: &mut Event, env: &mut Env) {
    match event {
        Event::Tick => {
            if let Page::LoadingPage(ref mut loading) = app_state.page {
                loading.tick(env);
            }
        }
        Event::Crossterm(crossterm_event) => {
            if let crossterm::event::Event::Key(key_event) = crossterm_event {
                if command_pallete_key_input_middleware(app_state, key_event, env) {
                    return;
                }
                if key_event.kind == KeyEventKind::Press {
                    match key_event.code {
                        KeyCode::Esc | KeyCode::Char('q') => app_state.running = false,
                        KeyCode::Char('c' | 'C') if key_event.modifiers == KeyModifiers::CONTROL => {
                            app_state.running = false
                        }
                        KeyCode::Enter => {
                            // Debounce example
                            let token = (env.gen_token)();
                            app_state.increment_token = Some(token.clone());
                            let sender = env.sender.clone();
                            env.fire_and_forget(async move {
                                tokio::time::sleep(Duration::from_secs(1)).await;
                                _ = sender.send(Action::Next(token).into());
                            });
                        },
                        KeyCode::Char('/') => {
                            app_state.command_pallette = Some(CommandPallette::new());
                        }
                        _ => {}
                    }
                }
            }
        }
        Event::App(action) => app_state_reduce(app_state, action, env),
    }
}

fn loading_page_reducer(state: &mut LoadingPage, action: &LoadingPageAction, env: &Env) {
    match action {
        LoadingPageAction::Finished => {
            state.is_finished = true;
            state.progress = 1.0;
            let sender = env.sender.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(300)).await;
                _ = sender.send(
                    Action::OpenPage(
                        Page::Main(MainPage::default())
                    ).into()
                );

            });
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            page: Page::LoadingPage(LoadingPage::default()),
            running: true,
            increment_token: None,
            command_pallette: None,
        }
    }
}

fn command_pallete_key_input_middleware(
    app_state: &mut AppState,
    key_event: &mut crossterm::event::KeyEvent,
    env: &Env,
) -> bool {
    match &mut app_state.command_pallette {
        Some(command_pallette) => {
            match key_event.code {
                KeyCode::Char('c' | 'C') if key_event.modifiers == KeyModifiers::CONTROL => {
                    false
                },
                KeyCode::Char('w' | 'W') if key_event.modifiers == KeyModifiers::CONTROL => {
                    let mut words = command_pallette.text_area_state.text.split(' ').collect::<Vec<&str>>();
                    if words.pop().is_some() {
                        let words = words.join(" ");
                        command_pallette.change_text(|text| *text = words);
                    }
                    true
                },
                KeyCode::Backspace => {
                    command_pallette.change_text(|text| {
                        text.pop();
                    });
                    true
                },
                KeyCode::Up => {
                    command_pallette.table_state.select_previous();
                    true
                },
                KeyCode::Down => {
                    command_pallette.table_state.select_next();
                    true
                },
                KeyCode::Esc => {
                    app_state.command_pallette = None;
                    true
                },
                KeyCode::Tab => {
                    if let Some(command) = command_pallette.command_to_complete() {
                        command_pallette.text_area_state.text = command.name().to_string();
                    }
                    true
                }
                KeyCode::Enter => {
                    if let Ok(command) = Command::try_from(command_pallette.text_area_state.text.as_str()) {
                        app_state.command_pallette = None;
                        _ = env.sender.send(Action::CommandSent(command).into());
                    } else if command_pallette.text_area_state.text.is_empty() {
                        if let Some(command) = command_pallette.selected_command() {
                            command_pallette.text_area_state.text = command.name().to_owned();
                        }
                    }
                    true
                },
                KeyCode::Char(char) if !char.is_control() => {
                    command_pallette.change_text(|text| text.push(char));
                    true
                },
                _ => false
            }
        }
        None => false,
    }
}