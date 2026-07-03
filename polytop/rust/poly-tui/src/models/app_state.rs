use std::time::Duration;
use std::assert;

use crossterm::event::{KeyCode, KeyEventKind, KeyModifiers};

use crate::env::Env;
use crate::event::{Event};
use crate::models::loading_page::{LoadingPage, LoadingPageAction, loading_page_reducer};
use crate::models::command::{CommandPallette, Command};
use crate::models::top_page::{TopPage, TopPageAction, top_page_reducer};

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
    Top(TopPage),
    Help(HelpPage),
}

#[derive(Clone, Debug)]
pub struct IntroPage {
}

#[derive(Clone, Debug)]
pub struct HelpPage {

}

#[derive(Clone, Debug)]
pub enum Action {
    Next(String),
    CommandClose,
    CommandSent(Command),
    LoadingPage(LoadingPageAction),
    TopPage(TopPageAction),
    OpenPage(Page),
    Quit,
}

impl Into<Event> for Action {
    fn into(self) -> Event {
        Event::App(self)
    }
}

pub fn app_state_reduce(app_state: &mut AppState, action: &mut Action, env: &Env) {
    match action {
        Action::Next(token) if app_state.increment_token.as_ref() == Some(token) => {
            //
        },
        Action::CommandClose => {
            app_state.command_pallette = None;
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
        },
        Action::TopPage(action) => {
            if let Page::Top(ref mut top_page) = app_state.page {
                top_page_reducer(top_page, &mut action.clone(), env);
            } else {
                assert!(false, "Action dispatched to non-loading page");
            }
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
                if let Page::Top(ref mut top_page) = app_state.page {
                    if top_page.key_input_middleware(key_event, env) {
                        return;
                    }
                }
                if let Some(command_pallette) = &mut app_state.command_pallette {
                    if command_pallette.command_pallete_key_input_middleware(key_event, env) {
                        return;
                    }
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
                                _ = sender.send(Action::LoadingPage(LoadingPageAction::Finished).into());
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
