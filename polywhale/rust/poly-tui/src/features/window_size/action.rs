#[derive(Clone, Debug)]
pub enum WindowSizeAction {
}

// After adding `Action::WindowSize(WindowSizeAction)` in features/app.rs:
//
// use crate::event::Event;
// use crate::features::app::Action;
//
// impl Into<Event> for WindowSizeAction {
//     fn into(self) -> Event {
//         Event::App(Action::WindowSize(self))
//     }
// }
