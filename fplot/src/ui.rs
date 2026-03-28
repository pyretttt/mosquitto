use ratatui::{
    Frame,
    widgets::Widget,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Gauge, List, ListItem, ListState, Paragraph},
};

use crate::app::AppState;
use crate::data;

pub fn draw(frame: &mut Frame, state: &mut AppState) {
    match &state.prices_feature.prices {
        data::crypto::PricesState::Initial | data::crypto::PricesState::Loading => {
            frame.render_widget(
                draw_loading(state.tick),
                frame.area().centered(
                Constraint::Percentage(50),
                Constraint::Length(3),
                )
            );
        }
        data::crypto::PricesState::Loaded(items) => {
            draw_prices(frame, frame.area(), items, &mut state.prices_feature.selected_index);
        }
        data::crypto::PricesState::PriceLoadFailed => {
            frame.render_widget(draw_error(), frame.area());
        }
    }
}

fn draw_loading(tick: u32) -> impl Widget {
    let progress = ((tick as f64 * 0.1).sin() + 1.0) / 2.0;
    let gauge = Gauge::default()
        .block(Block::bordered().title(" Loading "))
        .gauge_style(Style::default().fg(Color::Cyan).bg(Color::DarkGray))
        .ratio(progress)
        .label("Fetching prices…");
    gauge
}

fn draw_prices(
    frame: &mut Frame,
    area: Rect,
    prices: &[data::crypto::SymbolPrice],
    selected_index: &mut usize,
) {
    if !prices.is_empty() && *selected_index >= prices.len() {
        *selected_index = prices.len() - 1;
    }

    let items: Vec<ListItem> = prices
        .iter()
        .map(|sp| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{:<14}", sp.name),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" ${}", sp.price),
                    Style::default().fg(Color::Green),
                ),
            ]))
        })
        .collect();

    let status = format!(" {} symbols | ↑↓ scroll | q quit ", prices.len());

    let list = List::new(items)
        .block(
            Block::bordered()
                .title(" Symbols ")
                .title_alignment(Alignment::Center)
                .title_bottom(Line::from(status).centered()),
        )
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");

    let mut list_state = ListState::default().with_selected(Some(*selected_index));
    frame.render_stateful_widget(list, area, &mut list_state);
}

fn draw_error() -> impl Widget {
    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "⚠  Failed to load prices",
            Style::default()
                .fg(Color::Red)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "Retrying on next refresh cycle…",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let paragraph = Paragraph::new(lines)
        .block(
            Block::bordered()
                .title(" Error ")
                .title_alignment(Alignment::Center)
                .border_style(Style::default().fg(Color::Red)),
        )
        .alignment(Alignment::Center);
    paragraph
}
