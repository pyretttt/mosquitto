use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect, HorizontalAlignment, Margin};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Cell, Padding, Paragraph, Row, Table, TableState};

use crate::env::Env;
use crate::features::top_page::{Market, TopPage};
use crate::top_page_service::ActivityKind;

const BG: Color = Color::Black;
const BORDER: Color = Color::Rgb(0x33, 0x41, 0x55);
const ACCENT: Color = Color::LightGreen;
const POSITIVE: Color = Color::Rgb(0x22, 0xC5, 0x5E);
const NEGATIVE: Color = Color::Rgb(0xEF, 0x44, 0x44);
const WARNING: Color = Color::Rgb(0xF5, 0x9E, 0x0B);
const MUTED: Color = Color::DarkGray;
const HEADER: Color = Color::Yellow;

static SPACE: &'static str = " ";

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

    let [status_area, markets_area, lower_area, cmd_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Fill(3),
        Constraint::Fill(2),
        Constraint::Length(3),
    ])
    .areas(dashboard);

    render_status_bar(frame, status_area, top_page);
    render_top_markets(frame, markets_area, top_page);
    render_lower_panes(frame, lower_area, top_page);
    render_command_popup(frame, cmd_area, top_page);
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

fn render_top_markets(frame: &mut Frame, area: Rect, top_page: &TopPage) {
    let header = Row::new([
        header_cell("#"),
        header_cell("★"),
        header_cell("Market"),
        header_cell("Yes"),
        header_cell("No"),
        header_cell("24h"),
        header_cell("Move"),
        header_cell("Spread"),
    ]);

    let render_height = area.height.saturating_sub(3).max(1) as usize;

    let selected_index = top_page.markets_pane.table_state.selected().unwrap_or(0);

    let rows = top_page.markets_pane.markets_data.markets
        .iter()
        .enumerate()
        .map(|(index, market)|
            market_row(
                market,
                selected_index == index
            )
        );

    let table = Table::new(
        rows,
        [
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(28),
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(7),
        ],
    )
    .header(header)
    .block(
        pane_block(&top_page.markets_pane.title_label, Borders::ALL, top_page.current_pane == 1)
            .title_bottom(
                Line::from(top_page.markets_pane.footer_label.as_str())
                .centered(),
            ),
    )
    .highlight_symbol("▶ ");

    frame.render_stateful_widget(table, area, &mut top_page.markets_pane.table_state.clone());
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

fn render_command_popup(frame: &mut Frame, area: Rect, top_page: &TopPage) {
    let popup = &top_page.command_popup;
    let block = pane_block("Command Popup", Borders::ALL, false);
    let inner = block.inner(area);

    frame.render_widget(block, area);

    let [shortcuts_area, status_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(inner);

    frame.render_widget(
        Paragraph::new("f find   1-3 focus   ↑↓ move   b bookmarks/all   w bookmark   Tab chart   ? help")
            .style(Style::default().fg(MUTED).bg(BG)),
        shortcuts_area,
    );
    frame.render_widget(
        Paragraph::new(popup.status_label.as_str())
            .style(Style::default().fg(MUTED).bg(BG)),
        status_area,
    );
}

fn market_row<'a>(market: &'a Market, is_selected: bool) -> Row<'a> {
    Row::new([
        Cell::from(market.rank_label.as_str()).style(Style::default().fg(if is_selected { WARNING } else { MUTED })),
        Cell::from(market.bookmark_label).style(Style::default().fg(WARNING)),
        Cell::from(market.title.as_str()).style(Style::default().fg(if is_selected { WARNING } else { Color::White })),
        Cell::from(market.yes_label.as_str())
            .style(Style::default().fg(POSITIVE)),
        Cell::from(market.no_label.as_str())
            .style(Style::default().fg(NEGATIVE)),
        Cell::from(market.volume_label.as_str())
            .style(Style::default().fg(Color::White)),
        Cell::from(market.movement_label.as_str())
            .style(Style::default().fg(activity_kind_color(market.movement_kind))),
        Cell::from(market.spread_label.as_str())
            .style(Style::default().fg(MUTED)),
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
        Span::raw(SPACE),
        Span::styled(label, Style::default().fg(Color::White).bg(BG)),
        Span::raw(SPACE),
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