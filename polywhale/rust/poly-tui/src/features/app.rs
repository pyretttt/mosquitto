use std::time::Duration;
use std::assert;

use crossterm::event::{KeyCode, KeyEventKind, KeyModifiers};

use crate::env::Env;
use crate::event::{Event};
use crate::features::loading_page::{LoadingPage, LoadingPageAction, loading_page_reducer};
use crate::features::command::{CommandPallette, Command};
use crate::features::top_page::{TopPage, TopPageAction, top_page_reducer};
use crate::features::log_page::{LogPage};
use crate::features::window_size::{WindowSize, window_size_reducer, WindowSizeAction};
use ratatui::prelude::Size;

#[derive(Clone, Debug)]
pub struct AppState {
    pub page: Page,
    pub running: bool,
    pub increment_token: Option<String>,
    pub command_pallette: Option<CommandPallette>,
    pub overlay: Option<Overlay>,
}

#[derive(Clone, Debug)]
pub enum Overlay {
    Log(LogPage),
    WindowSize(WindowSize),
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
    OpenTopPage(TopPage),
    WindowSize(WindowSizeAction),
    OpenLogPage,
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
                    app_state.page = Page::Intro(IntroPage {});
                },
                Command::Log => {
                    app_state.overlay = Some(Overlay::Log(LogPage::new()));
                },
            }
            app_state.command_pallette = None;
        },
        Action::Next(_) => (),
        Action::Quit => app_state.running = false,
        Action::LoadingPage(action) => {
            if let Page::LoadingPage(ref mut loading) = app_state.page {
                loading_page_reducer(loading, action, env);
            } else {
                assert!(false, "Action dispatched to non-loading page {:?}", action);
            }
        },
        Action::OpenTopPage(page) => {
            app_state.page = Page::Top(std::mem::take(page));
        },
        Action::OpenLogPage => {
            app_state.overlay = Some(Overlay::Log(LogPage::new()));
        },
        Action::TopPage(action) => {
            if let Page::Top(ref mut top_page) = app_state.page {
                top_page_reducer(top_page, action, env);
            } else {
                assert!(false, "Action dispatched to non-loading page {:?}", action);
            }
        },
        Action::WindowSize(action) => {
            if let Some(Overlay::WindowSize(ref mut window_size)) = app_state.overlay {
                window_size_reducer(window_size, action, env);
            } else {
                assert!(false, "Action dispatched with no window size overlay {:?}", action);
            }
        },
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
            match crossterm_event {
                crossterm::event::Event::Key(key_event) => {
                    if let Some(command_pallette) = &mut app_state.command_pallette {
                        if command_pallette.command_pallete_key_input_middleware(key_event, env) {
                            return;
                        }
                    }

                    match &mut app_state.overlay {
                        Some(Overlay::Log(log_page)) => {
                            if log_page.key_input_middleware(key_event, env) {
                                return;
                            }
                        }
                        Some(Overlay::WindowSize(_)) => (),
                        _ => (),
                    }

                    match &mut app_state.page {
                        Page::Top(top_page) => {
                            if top_page.key_input_middleware(key_event, env) {
                                return;
                            }
                        }
                        _ => ()
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
                crossterm::event::Event::Resize(width, height) => {
                    log::info!(target: "app", "App: Resize: {:?}, {:?}", width, height);
                    let new_size = Size::new(*width, *height);
                    env.ui.window_size = new_size;

                    let invalid_size = new_size.width < env.ui.required_window_size.width || new_size.height < env.ui.required_window_size.height;
                    match &mut app_state.overlay {
                        Some(Overlay::WindowSize(window_size)) => {
                            if invalid_size {
                                window_size_reducer(
                                    window_size,
                                    &WindowSizeAction::Resize(new_size),
                                    env,
                                );
                            } else {
                                app_state.overlay = None;
                            }
                        }
                        _ if invalid_size => {
                            let mut window_size = WindowSize::new(env.ui.required_window_size);
                            window_size.current_size = new_size;
                            app_state.overlay = Some(Overlay::WindowSize(window_size));
                        },
                        _ => (),
                    }

                    match &mut app_state.page {
                        Page::Top(top_page) => {
                            top_page_reducer(top_page, &mut TopPageAction::Resize(new_size).into(), env);
                        }
                        _ => ()
                    }
                },
                _ => ()
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
            overlay: None,
        }
    }
}
