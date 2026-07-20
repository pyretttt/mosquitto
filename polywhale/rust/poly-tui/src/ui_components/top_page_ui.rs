use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect, HorizontalAlignment, Margin};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, BorderType, Borders, Cell, HighlightSpacing, Padding, Paragraph, Row, Table,
};

use crate::env::Env;
use crate::features::top_page::TopPage;
use crate::top_page_service::{ActivityKind, Event as PolyEvent, Market};

const BG: Color = Color::Black;
const BORDER: Color = Color::Rgb(0x33, 0x41, 0x55);
const ACCENT: Color = Color::LightGreen;
const POSITIVE: Color = Color::Rgb(0x22, 0xC5, 0x5E);
const NEGATIVE: Color = Color::Rgb(0xEF, 0x44, 0x44);
const WARNING: Color = Color::Rgb(0xF5, 0x9E, 0x0B);
const MUTED: Color = Color::DarkGray;
const HEADER: Color = Color::Yellow;

mod constants {
    pub static SPACE: &'static str = " ";
    pub static DOUBLE_SPACE: &'static str = "  ";
    pub static EMPTY: &'static str = "";
    pub static INTERMEDIATE_BAR: &'static str = "┃";
    pub static TOP_BAR: &'static str = "┳";
    pub static BOTTOM_BAR: &'static str = "┻";
    pub static TOP_TABLE_PAYLOAD_HEIGHT_ADDEND: u16 = 3;
}

pub fn top_page_ui(
    frame: &mut Frame,
    top_page: &TopPage,
    _env: &Env,
) {
    let outer = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(BORDER).bg(BG))
        .style(Style::default().bg(BG))
        .title_top(Line::from(top_page.left_title).left_aligned())
        .title_top(Line::from("").right_aligned());

    let dashboard = outer.inner(frame.area());
    frame.render_widget(outer, frame.area());

    let events_max_height = top_page.table_payload_height() + constants::TOP_TABLE_PAYLOAD_HEIGHT_ADDEND;

    let [status_area, events_area, lower_area, cmd_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Max(events_max_height),
        Constraint::Fill(2),
        Constraint::Length(3),
    ])
    .areas(dashboard);

    render_status_bar(frame, status_area, top_page);
    render_top_events(frame, events_area, top_page);
    render_lower_panes(frame, lower_area, top_page);
    key_bindings(frame, cmd_area, top_page);
}

fn render_status_bar(frame: &mut Frame, area: Rect, top_page: &TopPage) {
    let status = &top_page.status_pane;
    let online_icon = if status.is_online { " ✓ " } else { " ✗ " };
    let online_color = if status.is_online { POSITIVE } else { NEGATIVE };
    let online_label = if status.is_online { "net online" } else { "net offline" };
    let ws_label = if status.ws_live { "   ws live" } else { "   ws down" };
    let ws_color = if status.ws_live { ACCENT } else { NEGATIVE };

    let [left_area, right_area] = Layout::horizontal([
        Constraint::Fill(1),
        Constraint::Fill(1),
    ]).areas(area.inner(Margin::new(1, 0)));

    if let Some(error_msg) = top_page.error_msg.as_ref().map(|err| &err.left) {
        frame.render_widget(
            Paragraph::new(
                Line::from_iter([
                    Span::styled("⚠️ Error: ", Style::default().fg(WARNING)),
                    Span::styled(error_msg, Style::default().fg(NEGATIVE)),
                ])
            ),
            left_area
        );
    } else {
        frame.render_widget(Block::default().style(Style::default().bg(BG)), left_area);
    }

    frame.render_widget(
        Paragraph::new(Line::from_iter([
            Span::styled(online_icon, Style::default().fg(online_color)),
            Span::styled(online_label, Style::default().fg(Color::Gray)),
            Span::styled(ws_label, Style::default().fg(ws_color)),
            Span::styled(
                status.latency_label.as_str(),
                Style::default().fg(Color::White),
            ),
            Span::styled(
                status.refresh_label.as_str(),
                Style::default().fg(Color::Gray),
            ),
            Span::styled(
                status.mode_label.as_str(),
                Style::default().fg(Color::Gray),
            ),
        ])).alignment(HorizontalAlignment::Right),
        right_area,
    );
}

const fn events_column_constraints() -> [Constraint; 8] {
    [
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(28),
        Constraint::Length(6),
        Constraint::Length(6),
        Constraint::Length(7),
        Constraint::Length(7),
        Constraint::Length(7),
    ]
}

fn render_top_events(frame: &mut Frame, area: Rect, top_page: &TopPage) {
    let block = pane_block(&top_page.events_pane.title_label, Borders::ALL, top_page.current_pane == 1)
        .title_bottom(
            Line::from(top_page.events_pane.footer_label.as_str()).centered(),
        );
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let events = top_page.events_pane.event_slice();
    let selected = top_page
        .events_pane
        .table_state
        .selected()
        .unwrap_or(0)
        .min(events.len().saturating_sub(1));
    let markets_h = top_page.events_pane.markets_table_height();

    let rows = events.iter().enumerate().map(|(index, event)| {
        let is_selected = index == selected;
        event_row(
            event,
            is_selected,
            if is_selected { markets_h } else { 0 },
        )
    });

    let table = Table::new(rows, events_column_constraints())
        .header(Row::new([
            header_cell("#"),
            header_cell("★"),
            header_cell("Event"),
            header_cell("Yes"),
            header_cell("No"),
            header_cell("24h"),
            header_cell("Move"),
            header_cell("Spread"),
        ]));

    frame.render_stateful_widget(
        table,
        inner,
        &mut top_page.events_pane.table_state.clone(),
    );

    // Paint the markets table inside the expanded selected event row (below the
    // event title line). Pane height already grows by up to 6 when needed.
    if markets_h > 0 && !events.is_empty() {
        if let Some(markets_area) = markets_table_area(inner, selected as u16, markets_h) {
            render_markets_table(frame, markets_area, top_page);
        }
    }
}

/// Area inside the selected event row reserved for the nested markets table.
fn markets_table_area(table_area: Rect, selected_row: u16, markets_h: u16) -> Option<Rect> {
    let y = table_area
        .y
        .saturating_add(1) // header
        .saturating_add(selected_row) // rows above selection
        .saturating_add(1); // event title line
    if y >= table_area.bottom() || markets_h == 0 {
        assert!(false, "Incrorrect layout logic");
        return None;
    }
    let area = Rect {
        x: table_area.x,
        y,
        width: table_area.width,
        height: markets_h.min(table_area.bottom().saturating_sub(y)),
    };
    if area.height == 0 || area.width == 0 {
        assert!(false, "Incrorrect layout logic");
        None
    } else {
        Some(area)
    }
}

fn render_markets_table(frame: &mut Frame, area: Rect, top_page: &TopPage) {
    let Some(event) = top_page.events_pane.selected_event() else {
        assert!(false, "Impossible branch");
        return;
    };
    if event.markets.is_empty() {
        assert!(false, "Impossible branch");
        return;
    }

    let rows = event.markets.iter().enumerate().map(|(index, market)| {
        let bar = if index == 0 { constants::TOP_BAR } else if index == event.markets.len() - 1 { constants::BOTTOM_BAR } else { constants::INTERMEDIATE_BAR };
        market_row(market, bar)
    });
    let table = Table::new(rows, events_column_constraints())
        .highlight_spacing(HighlightSpacing::Always)
        .row_highlight_style(
            Style::default()
                .fg(WARNING)
                .add_modifier(Modifier::BOLD),
        );

    let mut state = top_page.events_pane.markets_table_state.clone();
    frame.render_stateful_widget(table, area, &mut state);
}

fn render_lower_panes(frame: &mut Frame, area: Rect, top_page: &TopPage) {
    let [selected_area, chart_activity_area] =
        Layout::horizontal([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)]).areas(area);

    render_selected_market(frame, selected_area, top_page);
    render_chart_activity(frame, chart_activity_area, top_page);
}

fn render_selected_market(frame: &mut Frame, area: Rect, top_page: &TopPage) {
    let market = &top_page.selected_market_pane.selected_market;
    let lines = [
        Line::styled(top_page.selected_market_pane.title, Style::default().fg(Color::White).bg(BG)),
        Line::from(""),
        Line::from_iter([
            Span::styled("yes  ", Style::default().fg(POSITIVE).bg(BG)),
            Span::styled(
                market.yes_label.as_str(),
                Style::default()
                    .fg(POSITIVE)
                    .bg(BG)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                market.yes_quotes_label.as_str(),
                Style::default().fg(MUTED).bg(BG),
            ),
        ]),
        Line::from_iter([
            Span::styled("no   ", Style::default().fg(NEGATIVE).bg(BG)),
            Span::styled(
                market.no_label.as_str(),
                Style::default()
                    .fg(NEGATIVE)
                    .bg(BG)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                market.no_quotes_label.as_str(),
                Style::default().fg(MUTED).bg(BG),
            ),
        ]),
        Line::from(""),
        metric_line("volume 24h    ", &market.volume_label),
        metric_line("liquidity     ", &market.liquidity_label),
        metric_line("open interest ", &market.open_interest_label),
        metric_line("end date      ", &market.end_date),
    ];

    frame.render_widget(
        Paragraph::new(lines.as_slice())
            .block(
                pane_block(
                    " [2] - Selected Market: summary ",
                    Borders::ALL,
                    top_page.current_pane == 2,
                )
                .padding(Padding::horizontal(1)),
            )
            .style(Style::default().bg(BG)),
        area,
    );
}

fn render_chart_activity(frame: &mut Frame, area: Rect, top_page: &TopPage) {
    let chart = &top_page.chart_activity_pane;
    let mut lines: Vec<Line> = chart
        .chart_activity
        .chart_lines
        .iter()
        .map(|line| Line::styled(line.as_str(), Style::default().fg(ACCENT).bg(BG)))
        .collect();
    lines.push(Line::from(""));

    for activity in &chart.chart_activity.activities {
        lines.push(activity_line(
            &activity.time,
            &activity.label,
            &activity.value,
            activity_kind_color(activity.kind),
        ));
    }

    frame.render_widget(
        Paragraph::new(lines)
            .block(pane_block(
                &chart.title_label,
                Borders::ALL,
                top_page.current_pane == 3,
            ))
            .style(Style::default().bg(BG)),
        area,
    );
}

fn key_bindings(frame: &mut Frame, area: Rect, top_page: &TopPage) {
    let popup = &top_page.command_popup;
    let block = pane_block("Keybindings: ", Borders::ALL, false);
    let inner = block.inner(area);

    frame.render_widget(block, area);

    let [shortcuts_area, status_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(inner);

    frame.render_widget(
        Paragraph::new("f find   1-3 focus   j/k move   →/enter markets   ←/esc events   ? help")
            .style(Style::default().fg(MUTED).bg(BG)),
        shortcuts_area,
    );
    frame.render_widget(
        Paragraph::new(popup.status_label.as_str())
            .style(Style::default().fg(MUTED).bg(BG)),
        status_area,
    );
}

fn event_row<'a>(event: &'a PolyEvent, is_selected: bool, markets_h: u16) -> Row<'a> {
    Row::new([
        Cell::from(event.rank_label.as_str()).style(Style::default().fg(if is_selected { WARNING } else { MUTED })),
        Cell::from(event.bookmark_label).style(Style::default().fg(WARNING)),
        Cell::from(event.title.as_str()).style(Style::default().fg(if is_selected { WARNING } else { Color::White })),
        Cell::from("—").style(Style::default().fg(MUTED)),
        Cell::from("—").style(Style::default().fg(MUTED)),
        Cell::from(event.volume_label.as_str()).style(Style::default().fg(Color::White)),
        Cell::from(event.markets_count_label.as_str()).style(Style::default().fg(MUTED)),
        Cell::from(constants::EMPTY).style(Style::default().fg(MUTED)),
    ])
    .height(1 + markets_h)
}

fn market_row<'a>(market: &'a Market, market_bar: &'static str) -> Row<'a> {
    Row::new([
        Cell::from(constants::EMPTY).style(Style::default().fg(MUTED)),
        Cell::from(market.bookmark_label).style(Style::default().fg(WARNING)),
        Cell::from(Line::from_iter([
            Span::styled(market_bar, Style::default().fg(MUTED)),
            Span::raw(constants::DOUBLE_SPACE),
            Span::styled(market.title.as_str(), Style::default().fg(Color::Gray)),
        ])),
        Cell::from(market.yes_label.as_str()).style(Style::default().fg(POSITIVE)),
        Cell::from(market.no_label.as_str()).style(Style::default().fg(NEGATIVE)),
        Cell::from(market.volume_label.as_str()).style(Style::default().fg(Color::White)),
        Cell::from(market.movement_label.as_str())
            .style(Style::default().fg(activity_kind_color(market.movement_kind))),
        Cell::from(market.spread_label.as_str()).style(Style::default().fg(MUTED)),
    ])
}

fn header_cell(label: &'static str) -> Cell<'static> {
    Cell::from(label).style(
        Style::default()
            .fg(HEADER)
            .bg(BG)
            .add_modifier(Modifier::BOLD),
    )
}

fn metric_line<'a>(label: &'static str, value: &'a str) -> Line<'a> {
    Line::from_iter([
        Span::styled(label, Style::default().fg(MUTED).bg(BG)),
        Span::styled(value, Style::default().fg(Color::White).bg(BG)),
    ])
}

fn activity_line<'a>(time: &'a str, label: &'a str, value: &'a str, value_color: Color) -> Line<'a> {
    Line::from_iter([
        Span::styled(time, Style::default().fg(MUTED).bg(BG)),
        Span::raw(constants::SPACE),
        Span::styled(label, Style::default().fg(Color::White).bg(BG)),
        Span::raw(constants::SPACE),
        Span::styled(value, Style::default().fg(value_color).bg(BG)),
    ])
}

fn pane_block<'a>(title: &'a str, borders: Borders, focused: bool) -> Block<'a> {
    let border_color = if focused { ACCENT } else { BORDER };

    Block::default()
        .borders(borders)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color))
        .title_top(Line::from(title).left_aligned())
}

fn activity_kind_color(kind: ActivityKind) -> Color {
    match kind {
        ActivityKind::Positive => POSITIVE,
        ActivityKind::Negative => NEGATIVE,
        ActivityKind::Accent => ACCENT,
        ActivityKind::Warning => WARNING,
        ActivityKind::Muted => MUTED,
    }
}
