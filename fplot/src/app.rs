use crate::data;
use crate::event::{AppEvent, Event, EventHandler, PriceEvent};
use crate::ui;
use crossterm::event::{KeyCode, KeyModifiers};
use ratatui::DefaultTerminal;
use std::sync::Arc;
use tokio::sync::mpsc;


pub struct Settings {
    pub price_refresh_interval_seconds: u32,
    pub tick_refresh_rate: f64,
}

pub struct Env {
    pub data_store: Arc<data::Store>,
    pub events: EventHandler,
    pub settings: Settings,
}

impl Env {
    pub fn event_sender(&self) -> mpsc::UnboundedSender<Event> {
        self.events.sender.clone()
    }

    pub fn get_send_event(&self) -> impl Fn(Event) -> () + use<> {
        let event_sender = self.event_sender().clone();
        move |event: Event| {
            let _ = event_sender.send(event);
        }
    }
}

impl Env {
    pub fn new(
        data_store: Arc<data::Store>,
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
    /// Counter
    pub tick: u32,
    /// Prices feature state.
    pub prices_feature: data::crypto::PricesFeatureState,
}

pub fn app_reducer<'a>(
    state: &'a mut AppState,
    action: &'a mut Event,
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
                    KeyCode::Up => env.events.send(AppEvent::UpKeyTapped),
                    KeyCode::Down => env.events.send(AppEvent::DownKeyTapped),
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
        state.prices_feature.prices = data::crypto::PricesState::Loading;
        let data_store = Arc::clone(&env.data_store);
        let send = env.get_send_event();
        tokio::spawn(async move {
            send(make_price_event(PriceEvent::PriceLoading));
            let result = data_store.get_prices().await;
            match result {
                Ok(prices) => send(make_price_event(PriceEvent::PricesLoaded(prices))),
                Err(_error) => send(make_price_event(PriceEvent::PriceLoadFailed)),
            }
        });
    }
    state
}

pub fn app_logic_reducer<'a>(
    state: &'a mut AppState,
    action: &'a mut AppEvent,
    _env: &'a mut Env
) -> &'a mut AppState {
    match action {
        AppEvent::Quit => {
            state.running = false;
        },
        AppEvent::UpKeyTapped => {
            if let data::crypto::PricesState::Loaded(_) = state.prices_feature.prices
                && state.prices_feature.selected_index > 0 {
                state.prices_feature.selected_index -= 1;
            }
        },
        AppEvent::DownKeyTapped => {
            if let data::crypto::PricesState::Loaded(ref prices) = state.prices_feature.prices
                && state.prices_feature.selected_index + 1 < prices.len() {
                state.prices_feature.selected_index += 1;
            }
        },
        AppEvent::Price(PriceEvent::PriceLoading) => {
            match state.prices_feature.prices {
                data::crypto::PricesState::Loading => {
                    state.prices_feature.prices = data::crypto::PricesState::Loading;
                }
                _ => {}
            }
        },
        AppEvent::Price(PriceEvent::PricesLoaded(prices)) => {
            state.prices_feature.last_update_tick_sec = state.tick;
            state.prices_feature.prices = data::crypto::PricesState::Loaded(std::mem::take(prices));
        },
        AppEvent::Price(PriceEvent::PriceLoadFailed) => {
            state.prices_feature.prices = data::crypto::PricesState::PriceLoadFailed;
        }
    }
    state
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            running: true,
            tick: 0,
            prices_feature: data::crypto::PricesFeatureState {
                prices: data::crypto::PricesState::Initial,
                last_update_tick_sec: 0,
                selected_index: 0,
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
        terminal.draw(|frame| ui::draw(frame, app_state))?;
        let mut event = env.events.next().await?;
        app_reducer(app_state, &mut event, env);
    }
    Ok(())
}


fn make_price_event(event: PriceEvent) -> Event {
    Event::App(AppEvent::Price(event))
}