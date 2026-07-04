use crossterm::event::{KeyCode, KeyModifiers};

use crate::env::Env;
use crate::features::app::{AppState, Action};
use ratatui::widgets::TableState;

static COMMANDS: &'static [Command] = &[
    Command::Help,
    Command::Quit,
    Command::Intro,
];

#[derive(Clone, Debug)]
pub struct CommandPallette {
    pub text_area_state: TextAreaState,
    pub table_state: TableState,
}

#[derive(Clone, Debug)]
pub struct TextAreaState {
    pub text: String,
    pub cursor_position: usize,
    pub input_placeholder: &'static str,
}

impl CommandPallette {
    pub fn new() -> Self {
        Self {
            text_area_state: TextAreaState {
                text: String::new(),
                cursor_position: 0,
                input_placeholder: "Type a command",
            },
            table_state: TableState::default().with_selected(0),
        }
    }

    pub fn available_commands(&self) -> impl Iterator<Item = &Command> {
        COMMANDS
            .iter()
            .filter(|command| command.name().starts_with(&self.text_area_state.text))
    }

    pub fn selected_command(&self) -> Option<&Command> {
        self.table_state
            .selected()
            .and_then(|index| self.available_commands().nth(index))
    }

    pub fn change_text(&mut self, modify: impl FnOnce(&mut String) -> ()) {
        modify(&mut self.text_area_state.text);
        self.table_state.select_first();
    }

    pub fn command_to_complete(&self) -> Option<&Command> {
        if self.text_area_state.text.len() == 0 {
            return None;
        }

        self.available_commands()
            .find(|command| command.name().starts_with(self.text_area_state.text.as_str()))
    }
}

#[derive(Clone, Debug)]
pub enum Command {
    Help,
    Quit,
    Intro,
}

impl Command {
    pub fn description(&self) -> &'static str {
        match self {
            Command::Help => "Show help",
            Command::Quit => "Quit the application",
            Command::Intro => "Show intro",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Command::Help => "help",
            Command::Quit => "quit",
            Command::Intro => "intro",
        }
    }
}

impl TryFrom<&str> for Command {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            val if val == Command::Help.name() => Ok(Command::Help),
            val if val == Command::Quit.name() => Ok(Command::Quit),
            val if val == Command::Intro.name() => Ok(Command::Intro),
            _ => Err(()),
        }
    }
}

impl CommandPallette {
    pub fn command_pallete_key_input_middleware(
        &mut self,
        key_event: &mut crossterm::event::KeyEvent,
        env: &Env,
    ) -> bool {
        match key_event.code {
            KeyCode::Char('c' | 'C') if key_event.modifiers == KeyModifiers::CONTROL => {
                false
            },
            KeyCode::Char('w' | 'W') if key_event.modifiers == KeyModifiers::CONTROL => {
                let mut words = self.text_area_state.text.split(' ').collect::<Vec<&str>>();
                if words.pop().is_some() {
                    let words = words.join(" ");
                    self.change_text(|text| *text = words);
                }
                true
            },
            KeyCode::Backspace => {
                self.change_text(|text| {
                    text.pop();
                });
                true
            },
            KeyCode::Up => {
                self.table_state.select_previous();
                true
            },
            KeyCode::Down => {
                self.table_state.select_next();
                true
            },
            KeyCode::Esc => {
                _ = env.sender.send(Action::CommandClose.into());
                true
            },
            KeyCode::Tab => {
                if let Some(command) = self.command_to_complete() {
                    self.text_area_state.text = command.name().to_string();
                }
                true
            }
            KeyCode::Enter => {
                if let Ok(command) = Command::try_from(self.text_area_state.text.as_str()) {
                    _ = env.sender.send(Action::CommandClose.into());
                    _ = env.sender.send(Action::CommandSent(command).into());
                } else if self.text_area_state.text.is_empty() {
                    if let Some(command) = self.selected_command() {
                        self.text_area_state.text = command.name().to_owned();
                    }
                }
                true
            },
            KeyCode::Char(char) if !char.is_control() => {
                self.change_text(|text| text.push(char));
                true
            },
            _ => false
        }
    }
}