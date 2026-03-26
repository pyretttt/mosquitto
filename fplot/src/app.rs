use crate::event::{AppEvent, Event, EventHandler};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::DefaultTerminal;


pub struct Settings {
    pub price_refresh_interval_seconds: u32,
    pub tick_refresh_rate: f64,
}

pub static SETTINGS: Settings = Settings {
    price_refresh_interval_seconds: 10,
    tick_refresh_rate: 30.0,
};

struct Env {
    data_store: crate::data_store::DataStore,
}

impl Env {
    pub fn new(
        data_store: crate::data_store::DataStore,
    ) -> Self {
        Self {
            data_store: data_store,
        }
    }
}

static ENV: std::sync::LazyLock<Env> = std::sync::LazyLock::new(|| Env::new(
    crate::data_store::DataStore::new(
        reqwest::Client::new()
    )
));

/// Application.
#[derive(Debug)]
pub struct App {
    /// Is the application running?
    pub running: bool,
    /// Counter.
    pub counter: u8,
    /// Event handler.
    pub events: EventHandler,
    /// Counter
    pub tick: u32,
}

impl Default for App {
    fn default() -> Self {
        Self {
            running: true,
            counter: 0,
            events: EventHandler::new(),
            tick: 0
        }
    }
}

impl App {
    /// Constructs a new instance of [`App`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Run the application's main loop.
    pub async fn run(mut self, mut terminal: DefaultTerminal) -> color_eyre::Result<()> {
        while self.running {
            terminal.draw(|frame| frame.render_widget(&self, frame.area()))?;

            match self.events.next().await? {
                Event::Tick => self.tick(),
                Event::Crossterm(event) => match event {
                    crossterm::event::Event::Key(key_event)
                        if key_event.kind == crossterm::event::KeyEventKind::Press =>
                    {
                        self.handle_key_events(key_event)?
                    }
                    _ => {
                        println!("unexpected event: {:?}", event);
                    }
                },
                Event::App(app_event) => match app_event {
                    AppEvent::Increment => self.increment_counter(),
                    AppEvent::Decrement => self.decrement_counter(),
                    AppEvent::Quit => self.quit(),
                },
            }
        }
        Ok(())
    }

    /// Handles the key events and updates the state of [`App`].
    pub fn handle_key_events(&mut self, key_event: KeyEvent) -> color_eyre::Result<()> {
        match key_event.code {
            KeyCode::Esc | KeyCode::Char('q') => self.events.send(AppEvent::Quit),
            KeyCode::Char('c' | 'C') if key_event.modifiers == KeyModifiers::CONTROL => {
                self.events.send(AppEvent::Quit)
            }
            KeyCode::Right => self.events.send(AppEvent::Increment),
            KeyCode::Left => self.events.send(AppEvent::Decrement),
            // Other handlers you could add here.
            _ => {}
        }
        Ok(())
    }

    /// Handles the tick event of the terminal.
    ///
    /// The tick event is where you can update the state of your application with any logic that
    /// needs to be updated at a fixed frame rate. E.g. polling a server, updating an animation.
    pub fn tick(&mut self) {
        self.tick = self.tick.wrapping_add(1);

        let tick_seconds = self.tick / SETTINGS.tick_refresh_rate as u32;
        if tick_seconds % SETTINGS.price_refresh_interval_seconds == 0 {
            tokio::spawn(async {
                let prices = ENV.data_store.get_prices().await;
                match prices {
                    Ok(prices) => {
                    }
                    Err(e) => {
                    }
                }
            });
        }
    }

    /// Set running to false to quit the application.
    pub fn quit(&mut self) {
        self.running = false;
    }

    pub fn increment_counter(&mut self) {
        self.counter = self.counter.saturating_add(1);
    }

    pub fn decrement_counter(&mut self) {
        self.counter = self.counter.saturating_sub(1);
    }
}
