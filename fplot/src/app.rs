use crate::event::{AppEvent, Event, EventHandler};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::DefaultTerminal;
use crate::data;

pub struct Settings {
    pub price_refresh_interval_seconds: u32,
    pub tick_refresh_rate: f64,
}

pub struct Env {
    pub data_store: data::DataStore,
    pub events: EventHandler,
    pub settings: Settings,
    pub terminal: DefaultTerminal,
}

impl Env {
    pub fn new(
        data_store: data::DataStore,
        events: EventHandler,
        settings: Settings,
        terminal: DefaultTerminal
    ) -> Self {
        Self {
            data_store: data_store,
            events: events,
            settings: settings,
            terminal: terminal,
        }
    }
}

/// Application.
#[derive(Debug)]
pub struct AppState {
    /// Is the application running?
    pub running: bool,
    /// Counter.
    pub counter: u8,
    /// Counter
    pub tick: u32,
    /// Prices feature state.
    pub prices_feature: data::crypto_market::PricesFeatureState,
}

pub fn app_reducer<'a>(
    state: &'a mut AppState,
    action: &'a Event,
    env: &'a Env
) -> &'a mut AppState {
    match action {
        Event::Tick => {
            state.tick = state.tick.wrapping_add(1);
        },
        Event::Crossterm(event) => match event {
            crossterm::event::Event::Key(key_event)
                if key_event.kind == crossterm::event::KeyEventKind::Press =>
            {
                match key_event.code {
                    KeyCode::Esc | KeyCode::Char('q') => env.events.send(AppEvent::Quit),
                    KeyCode::Char('c' | 'C') if key_event.modifiers == KeyModifiers::CONTROL => {
                        env.events.send(AppEvent::Quit)
                    }
                    KeyCode::Right => env.events.send(AppEvent::Increment),
                    KeyCode::Left => env.events.send(AppEvent::Decrement),
                    // Other handlers you could add here.
                    _ => {}
                }
            }
            _ => {
                println!("unexpected event: {:?}", event);
            }
            
        },
        Event::App(app_event) => {
            state = app_logic_reducer(state, app_event, env);
        },
    }

    state
}

pub fn app_logic_reducer<'a>(
    state: &'a mut AppState, 
    action: &'a AppEvent, 
    _env: &'a Env
) -> &'a mut AppState {
    match action {
        AppEvent::Increment => {
            state.counter = state.counter.wrapping_add(1);
        },
        AppEvent::Decrement => {
            state.counter = state.counter.wrapping_sub(1);
        },
        AppEvent::Quit => {
            state.running = false;
        },
    }
    state
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            running: true,
            counter: 0,
            tick: 0,
            prices_feature: data::crypto_market::PricesFeatureState {
                prices: data::crypto_market::PricesState::Initial,
                last_update_tick_sec: 0,
            },
        }
    }
}

pub async fn run(app_state:& mut AppState, env: &Env) -> color_eyre::Result<()> {
    while app_state.running {
        env.terminal.draw(|frame| frame.render_widget(app_state as &AppState, frame.area()))?;
        let event = env.events.next().await?;
        app_state = app_reducer(&mut app_state, &event, env);
    }
    Ok(())
}