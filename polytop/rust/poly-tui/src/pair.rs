
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Pair<L, R> {
    pub left: L,
    pub right: R,
}

impl<L, R> Pair<L, R> {
    pub fn new(left: L, right: R) -> Self {
        Self { left, right }
    }
}