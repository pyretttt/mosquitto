use std::time::Duration;

use iocraft::prelude::*;

use crate::config::config;
use crate::env::Env;
use crate::models::app_state::{app_state_reduce, Action, AppState, IntroPage, Page};

#[component]
pub fn App(mut hooks: Hooks) -> impl Into<AnyElement<'static>> {
    let (width, height) = hooks.use_terminal_size();
    let mut app_state = hooks.use_state(|| AppState {
        page: Page::Intro(IntroPage {
            title: "Polytop - Polymarket Monitor".to_owned(),
            text: "Hello, world!".to_owned(),
        }),
        counter: 0,
    });
    let env = hooks.use_ref(Env::new);
    let mut system = hooks.use_context_mut::<SystemContext>();
    let mut should_exit = hooks.use_state(|| false);

    hooks.use_terminal_events({ move |event|
        match event {
            TerminalEvent::Key(KeyEvent { code, kind, .. }) if kind != KeyEventKind::Release => {
                if matches!(code, KeyCode::Char('q')) {
                    should_exit.set(true);
                }
            }
            _ => {}
        }
    });

    hooks.use_future({
        let env = env.clone();
        async move {
            let tick_rate = config().tick_rate;
            loop {
                smol::Timer::after(Duration::from_secs_f64(1.0 / tick_rate)).await;
                let mut state = app_state.write();
                app_state_reduce(&mut state, Action::Next, &env.read());
            }
        }
    });

    if should_exit.get() {
        system.exit();
    }

    let Page::Intro(intro) = &app_state.read().page;

    element! {
        View(
            width,
            height,
            flex_direction: FlexDirection::Column,
            padding_left: 2,
            padding_top: 1,
        ) {
            Text(
                content: intro.title.clone(),
                color: Color::Magenta,
                weight: Weight::Bold,
            )
            View(margin_top: 1) {
                Text(content: intro.text.clone())
            }
            View(margin_top: 1) {
                Text(content: format!("counter: {}", app_state.read().counter))
            }
            View(margin_top: 2) {
                Text(content: "Press q to quit", color: Color::DarkGrey)
            }
        }
    }
}
