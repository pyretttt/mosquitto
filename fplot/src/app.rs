use crate::data;
use crate::event::{AppEvent, Event, EventHandler, PriceEvent};
use crossterm::event::{KeyCode, KeyModifiers};
use ratatui::DefaultTerminal;
use std::sync::Arc;
use tokio::sync::mpsc;


pub struct Settings {
    pub price_refresh_interval_seconds: u32,
    pub tick_refresh_rate: f64,
}

pub struct Env {
    pub data_store: Arc<data::DataStore>,
    pub events: EventHandler,
    pub settings: Settings,
}

impl Env {
    pub fn event_sender(&self) -> mpsc::UnboundedSender<Event> {
        self.events.sender.clone()
    }
}

impl Env {
    pub fn new(
        data_store: Arc<data::DataStore>,
        events: EventHandler,
        settings: Settings,
    ) -> Self {
        Self {
            data_store,
            events,
            settings,
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
    env: &'a mut Env,
) -> &'a mut AppState {
    match action {
        Event::Tick => {
            tick_reducer(state, env);
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
            app_logic_reducer(state, app_event, env);
        },
    }

    state
}

pub fn tick_reducer<'a>(
    state: &'a mut AppState,
    env: &'a mut Env,
) -> &'a mut AppState {
    state.tick = state.tick.wrapping_add(1);

    let tick_sec = state.tick / env.settings.tick_refresh_rate as u32;
    if tick_sec % env.settings.price_refresh_interval_seconds == 0 {
        state.prices_feature.prices = data::crypto_market::PricesState::Loading;
        let data_store = Arc::clone(&env.data_store);
        let event_sender = env.event_sender();
        tokio::spawn(async move {
            event_sender.send(Event::App(PriceEvent::PriceLoading));
            let prices = data_store.get_prices().await;
            match prices {
                Ok(prices) => event_sender.send(Event::App(PriceEvent::PricesLoaded(prices))),
                Err(error) => event_sender.send(Event::App(PriceEvent::PriceLoadFailed)),
            }
        });
    }
    state
}

pub fn app_logic_reducer<'a>(
    state: &'a mut AppState, 
    action: &'a AppEvent, 
    _env: &'a mut Env
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
        AppEvent::Price(PriceEvent::PriceLoading) => {
            match state.prices_feature.prices {
                data::crypto_market::PricesState::Loading => {
                    state.prices_feature.prices = data::crypto_market::PricesState::Loading;
                }
                _ => {}
            }
        },
        AppEvent::Price(PriceEvent::PricesLoaded(prices)) => {
            state.prices_feature.prices = data::crypto_market::PricesState::Loaded(prices);
        },
        AppEvent::Price(PriceEvent::PriceLoadFailed) => {
            state.prices_feature.prices = data::crypto_market::PricesState::PriceLoadFailed;
        }
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

pub async fn run(
    app_state: &mut AppState,
    env: &mut Env,
    terminal: &mut DefaultTerminal,
) -> color_eyre::Result<()> {
    while app_state.running {
        terminal.draw(|frame| frame.render_widget(app_state as &AppState, frame.area()))?;
        let event = env.events.next().await?;
        app_reducer(app_state, &event, env);
    }
    Ok(())
}
