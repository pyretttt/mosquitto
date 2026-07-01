use std::time::{SystemTime, UNIX_EPOCH};

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect, Margin};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Cell, Padding, Paragraph, Row, Table, TableState};

use crate::env::Env;
use crate::models::app_state::AppState;
use crate::models::top_page::TopPage;

const BG: Color = Color::Black;
const BORDER: Color = Color::Rgb(0x33, 0x41, 0x55);
const ACCENT: Color = Color::LightGreen;
const POSITIVE: Color = Color::Rgb(0x22, 0xC5, 0x5E);
const NEGATIVE: Color = Color::Rgb(0xEF, 0x44, 0x44);
const WARNING: Color = Color::Rgb(0xF5, 0x9E, 0x0B);
const MUTED: Color = Color::DarkGray;
const HEADER: Color = Color::Yellow;

pub fn top_page_ui(
    frame: &mut Frame,
    _app_state: &AppState,
    top_page: &TopPage,
    _env: &Env,
) {
    let outer = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(BORDER).bg(BG))
        .style(Style::default().bg(BG))
        .title_top(Line::from(" POLYTOP ").left_aligned())
        .title_top(Line::from(clock_hms()).right_aligned());

    let dashboard = outer.inner(frame.area());
    frame.render_widget(outer, frame.area());

    let [status_area, markets_area, lower_area, cmd_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Fill(3),
        Constraint::Fill(2),
        Constraint::Length(3),
    ])
    .areas(dashboard);

    render_status_bar(frame, status_area);
    render_top_markets(frame, markets_area, top_page);
    render_lower_panes(frame, lower_area, top_page);
    render_command_popup(frame, cmd_area);
}

fn render_status_bar(frame: &mut Frame, area: Rect) {
    frame.render_widget(
        Paragraph::new(Line::from_iter([
            Span::styled(" ✓ ", Style::default().fg(POSITIVE)),
            Span::styled("net online", Style::default().fg(Color::Gray)),
            Span::styled("   ws live", Style::default().fg(ACCENT)),
            Span::styled("   latency 42ms", Style::default().fg(Color::White)),
            Span::styled("   refresh 500ms", Style::default().fg(Color::Gray)),
            Span::styled("   mode observe", Style::default().fg(Color::Gray)),
        ])),
        area,
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

    let rows = [
        market_row(
            "1",
            "★",
            "Will BTC hit 100k in 2026?",
            "63¢",
            "38¢",
            "842k",
            "+4¢",
            "2¢",
        ),
        market_row("2", "", "Fed cuts rates by Sep?", "41¢", "60¢", "611k", "-2¢", "3¢"),
        market_row("3", "", "Lakers win tonight?", "55¢", "46¢", "570k", "+1¢", "2¢"),
        market_row(
            "4",
            "",
            "ETH ETF inflows above $1B?",
            "72¢",
            "29¢",
            "510k",
            "+6¢",
            "4¢",
        ),
        market_row(
            "5",
            "★",
            "Trump wins popular vote?",
            "49¢",
            "52¢",
            "421k",
            "-1¢",
            "2¢",
        ),
        market_row("6", "", "CPI below forecast?", "36¢", "65¢", "390k", "-5¢", "5¢"),
        market_row("7", "", "SpaceX launch this week?", "83¢", "18¢", "311k", "+8¢", "3¢"),
        market_row("8", "", "Oil closes above $90?", "22¢", "79¢", "280k", "-3¢", "4¢"),
    ];

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
        pane_block(" [1] - Top Markets: all ", Borders::ALL, top_page.current_pane == 1)
            .title_bottom(
                Line::from(" 8 markets | ↑↓ move | b bookmarks/all | w bookmark ").centered(),
            ),
    )
    .row_highlight_style(
        Style::default()
            .bg(Color::DarkGray)
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    )
    .highlight_symbol("▶ ");

    let mut state = TableState::default().with_selected(Some(0));
    frame.render_stateful_widget(table, area, &mut state);
}

fn render_lower_panes(frame: &mut Frame, area: Rect, top_page: &TopPage) {
    let [selected_area, chart_activity_area] =
        Layout::horizontal([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)]).areas(area);

    render_selected_market(frame, selected_area, top_page);
    render_chart_activity(frame, chart_activity_area, top_page);
}

fn render_selected_market(frame: &mut Frame, area: Rect, top_page: &TopPage) {
    let lines = [
        Line::styled(
            "Will BTC hit 100k in 2026?",
            Style::default().fg(Color::White).bg(BG),
        ),
        Line::from(""),
        Line::from_iter([
            Span::styled("yes  ", Style::default().fg(POSITIVE).bg(BG)),
            Span::styled(
                "63¢",
                Style::default()
                    .fg(POSITIVE)
                    .bg(BG)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "  bid 62 / ask 64   spread 2¢",
                Style::default().fg(MUTED).bg(BG),
            ),
        ]),
        Line::from_iter([
            Span::styled("no   ", Style::default().fg(NEGATIVE).bg(BG)),
            Span::styled(
                "38¢",
                Style::default()
                    .fg(NEGATIVE)
                    .bg(BG)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  bid 37 / ask 39", Style::default().fg(MUTED).bg(BG)),
        ]),
        Line::from(""),
        metric_line("volume 24h", "$842.1k"),
        metric_line("liquidity", "$184.3k"),
        metric_line("open interest", "$2.4M"),
        metric_line("end date", "2026-12-31"),
    ];

    frame.render_widget(
        Paragraph::new(lines.as_slice())
            .block(
                pane_block(" [2] - Selected Market: summary ", Borders::ALL, top_page.current_pane == 2)
                .padding(Padding::horizontal(1))
            )
            .style(Style::default().bg(BG)),
        area,
    );
}

fn render_chart_activity(frame: &mut Frame, area: Rect, top_page: &TopPage) {
    let lines = [
        Line::styled(
            " 70¢ ┤                     ╭╮",
            Style::default().fg(ACCENT).bg(BG),
        ),
        Line::styled(
            " 65¢ ┤              ╭──────╯╰─╮",
            Style::default().fg(ACCENT).bg(BG),
        ),
        Line::styled(
            " 60¢ ┤      ╭───────╯         ╰╮",
            Style::default().fg(ACCENT).bg(BG),
        ),
        Line::styled(
            " 55¢ ┤ ╭────╯                  ╰─",
            Style::default().fg(ACCENT).bg(BG),
        ),
        Line::styled(
            "     └────────────────────────────",
            Style::default().fg(MUTED).bg(BG),
        ),
        Line::from(""),
        activity_line("17:42", "price", "+4¢", POSITIVE),
        activity_line("17:41", "best bid", "62¢", ACCENT),
        activity_line("17:40", "trade", "219 @45¢", WARNING),
        activity_line("17:39", "spread", "2¢", MUTED),
    ];

    frame.render_widget(
        Paragraph::new(lines.as_slice())
            .block(pane_block(" [3] - Chart + Activity ", Borders::ALL, top_page.current_pane == 3))
            .style(Style::default().bg(BG)),
        area,
    );
}

fn render_command_popup(frame: &mut Frame, area: Rect) {
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
        Paragraph::new("Filter: politics volume>100k sort:move                               NORMAL")
            .style(Style::default().fg(MUTED).bg(BG)),
        status_area,
    );
}

fn header_cell(label: &'static str) -> Cell<'static> {
    Cell::from(label).style(
        Style::default()
            .fg(HEADER)
            .bg(BG)
            .add_modifier(Modifier::BOLD),
    )
}

#[allow(clippy::too_many_arguments)]
fn market_row<'a>(
    rank: &'a str,
    bookmark: &'a str,
    market: &'a str,
    yes: &'a str,
    no: &'a str,
    volume: &'a str,
    movement: &'a str,
    spread: &'a str,
) -> Row<'a> {
    let move_color = if movement.starts_with('+') {
        POSITIVE
    } else if movement.starts_with('-') {
        NEGATIVE
    } else {
        MUTED
    };

    Row::new([
        Cell::from(rank).style(Style::default().fg(MUTED).bg(BG)),
        Cell::from(bookmark).style(Style::default().fg(WARNING).bg(BG)),
        Cell::from(market).style(Style::default().fg(Color::White).bg(BG)),
        Cell::from(yes).style(Style::default().fg(POSITIVE).bg(BG)),
        Cell::from(no).style(Style::default().fg(NEGATIVE).bg(BG)),
        Cell::from(volume).style(Style::default().fg(Color::White).bg(BG)),
        Cell::from(movement).style(Style::default().fg(move_color).bg(BG)),
        Cell::from(spread).style(Style::default().fg(MUTED).bg(BG)),
    ])
}

fn metric_line(label: &'static str, value: &'static str) -> Line<'static> {
    Line::from_iter([
        Span::styled(format!("{label:<14}"), Style::default().fg(MUTED).bg(BG)),
        Span::styled(value, Style::default().fg(Color::White).bg(BG)),
    ])
}

fn activity_line<'a>(
    time: &'a str,
    label: &'a str,
    value: &'a str,
    value_color: Color,
) -> Line<'a> {
    Line::from_iter([
        Span::styled(time, Style::default().fg(MUTED).bg(BG)),
        Span::styled(format!(" {label:<8} "), Style::default().fg(Color::White).bg(BG)),
        Span::styled(value, Style::default().fg(value_color).bg(BG)),
    ])
}

fn pane_block(title: &'static str, borders: Borders, focused: bool) -> Block<'static> {
    let border_color = if focused { ACCENT } else { BORDER };

    Block::default()
        .borders(borders)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color))
        .title_top(Line::from(title).left_aligned())
}

fn clock_hms() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() % 86_400)
        .unwrap_or(0);

    format!(
        " {:02}:{:02}:{:02} ",
        seconds / 3_600,
        (seconds % 3_600) / 60,
        seconds % 60
    )
}