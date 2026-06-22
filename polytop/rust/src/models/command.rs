static COMMANDS: &'static [Command] = &[
    Command::Help,
    Command::Quit,
];

#[derive(Clone, Debug)]
pub struct CommandPallette {
    pub input_text: String,
    pub input_placeholder: &'static str,
    pub commands: &'static [Command],
}

impl CommandPallette {
    pub fn new() -> Self {
        Self {
            input_text: String::new(),
            input_placeholder: "Type a command",
            commands: COMMANDS,
        }
    }

    pub fn available_commands(&self) -> impl Iterator<Item = &Command> {
        COMMANDS
            .iter()
            .filter(|command| command.name().starts_with(&self.input_text))
    }
}

#[derive(Clone, Debug)]
pub enum Command {
    Help,
    Quit,
    Intro,
}

impl Command {
    pub fn description(&self) -> &str {
        match self {
            Command::Help => "Show help",
            Command::Quit => "Quit the application",
            Command::Intro => "Show intro",
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Command::Help => "help",
            Command::Quit => "quit",
            Command::Intro => "intro",
        }
    }
}