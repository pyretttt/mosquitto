pub mod config;
pub mod models;
pub mod ui;
pub mod env;

use iocraft::prelude::*;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    smol::block_on(element!(ui::App).fullscreen())?;
    Ok(())
}
