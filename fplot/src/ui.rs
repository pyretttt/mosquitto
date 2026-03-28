use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Cell, Paragraph, Row, Table, TableState},
};

use crate::app::AppState;
use crate::data;

pub fn draw(frame: &mut Frame, state: &mut AppState) {
    let [status_area, table_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(0),
    ]).areas(frame.area());

    frame.render_widget(
        status_bar(&state.prices_feature),
        status_area,
    );
    draw_prices_table(
        frame,
        table_area,
        &state.prices_feature.prices,
        &mut state.prices_feature.selected_index,
    );
}

fn status_bar(feature: &data::crypto::PricesFeatureState) -> Paragraph<'_> {
    match &feature.loading {
        data::crypto::PricesLoadingState::Loading => Paragraph::new(Line::from(vec![
            Span::styled(" ⟳ ", Style::default().fg(Color::Cyan)),
            Span::styled(
                "Loading prices…",
                Style::default().fg(Color::DarkGray),
            ),
        ])),
        data::crypto::PricesLoadingState::PriceLoadFailed => Paragraph::new(Line::from(vec![
            Span::styled(" ⚠ ", Style::default().fg(Color::Red)),
            Span::styled(
                "Failed to load prices",
                Style::default()
                    .fg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            ),
        ])),
        data::crypto::PricesLoadingState::Idle => Paragraph::new(Line::from(vec![
            Span::styled(" ✓ ", Style::default().fg(Color::Green)),
            Span::styled(
                format!("{} symbols loaded", feature.prices.len()),
                Style::default().fg(Color::DarkGray),
            ),
        ])),
    }
}

fn price_row(symbol: &data::crypto::SymbolPrice) -> Row<'_> {
    Row::new(vec![
        Cell::from(symbol.name.as_str())
            .style(Style::default().fg(Color::White)),
        Cell::from(format!("${:.2}", symbol.price))
            .style(Style::default().fg(Color::Green)),
    ])
}

fn draw_prices_table(
    frame: &mut Frame,
    area: Rect,
    prices: &[data::crypto::SymbolPrice],
    selected_index: &mut usize,
) {
    let header = Row::new(vec![
        Cell::from("Symbol").style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Cell::from("Price").style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
    ]);

    let rows: Vec<Row> = prices.iter().map(price_row).collect();
    let status = format!(" {} symbols | ↑↓ scroll | q quit ", prices.len());

    let table = Table::new(rows, [Constraint::Min(14), Constraint::Min(14)])
        .header(header)
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

    let mut table_state = TableState::default().with_selected(Some(*selected_index));
    frame.render_stateful_widget(table, area, &mut table_state);
}
