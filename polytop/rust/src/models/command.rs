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

    pub fn command_to_complete(&self) -> Option<&Command> {
        if self.input_text.len() == 0 { return None; }

        self.available_commands().find(|command| command.name().starts_with(self.input_text.as_str()))
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

impl TryFrom<&str> for Command {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            val if val == Command::Help.name() => Ok(Command::Help),
            val if val == Command::Quit.name()=> Ok(Command::Quit),
            val if val == Command::Intro.name() => Ok(Command::Intro),
            _ => Err(()),
        }
    }
}