
pub mod reducer {
    pub trait Reducer<State, Action, Env>: Fn(State, Action, Env) -> State {}

    impl<F, State, Action, Env> Reducer<State, Action, Env> for F
    where
    F: Fn(State, Action, Env) -> State,
    {}
}