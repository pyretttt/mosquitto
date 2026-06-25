use std::time::{SystemTime, UNIX_EPOCH};

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};

use crate::env::Env;
use crate::models::app_state::AppState;
use crate::models::top_page::TopPage;

const BG: Color = Color::Rgb(0x0B, 0x10, 0x20);
const BORDER: Color = Color::Rgb(0x33, 0x41, 0x55);
const ACCENT: Color = Color::Rgb(0x22, 0xD3, 0xEE);
const MUTED: Color = Color::Rgb(0x94, 0xA3, 0xB8);

const LEFT_COLUMN: [Constraint; 2] = [Constraint::Ratio(35, 100), Constraint::Ratio(65, 100)];

pub fn top_page_ui(
    frame: &mut Frame,
    _app_state: &AppState,
    _top_page: &TopPage,
    _env: &Env,
) {
    let outer = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(BORDER).bg(BG))
        .style(Style::default().bg(BG))
        .title_top(Line::from(" POLYTOP ").left_aligned())
        .title_top(Line::from(clock_hms()).right_aligned());

    frame.render_widget(outer.clone(), frame.area());
    let dashboard = outer.inner(frame.area());

    let [header_area, l1_area, l2_area, l3_area, cmd_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Fill(37500),
        Constraint::Fill(31250),
        Constraint::Fill(15625),
        Constraint::Length(3),
    ])
    .areas(dashboard);

    frame.render_widget(
        Paragraph::new("net: online   ws: live   latency: 42ms   refresh: 500ms   mode: observe")
            .style(Style::default().fg(MUTED).bg(BG)),
        header_area,
    );

    render_l1(frame, l1_area);
    render_l2(frame, l2_area);
    render_l3(frame, l3_area);
    render_command_popup(frame, cmd_area);
}

fn render_l1(frame: &mut Frame, area: Rect) {
    let [snapshot_area, markets_area] = Layout::horizontal(LEFT_COLUMN).areas(area);

    render_pane(
        frame,
        snapshot_area,
        "[1] - Market Snapshot",
        Borders::ALL,
        false,
    );
    render_pane(
        frame,
        markets_area,
        "[2] - Top Markets",
        Borders::TOP | Borders::RIGHT | Borders::BOTTOM,
        false,
    );
}

fn render_l2(frame: &mut Frame, area: Rect) {
    let [watchlist_area, right_area] = Layout::horizontal(LEFT_COLUMN).areas(area);
    let [selected_area, activity_area] =
        Layout::horizontal([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)]).areas(right_area);

    render_pane(
        frame,
        watchlist_area,
        "[3] - Watchlist",
        Borders::LEFT | Borders::RIGHT | Borders::BOTTOM,
        false,
    );
    render_pane(
        frame,
        selected_area,
        "[4] - Selected Market",
        Borders::RIGHT | Borders::BOTTOM,
        false,
    );
    render_pane(
        frame,
        activity_area,
        "[5] - Activity",
        Borders::RIGHT | Borders::BOTTOM,
        false,
    );
}

fn render_l3(frame: &mut Frame, area: Rect) {
    render_pane(
        frame,
        area,
        "[6] - Order Book / Depth",
        Borders::LEFT | Borders::RIGHT | Borders::BOTTOM,
        false,
    );
}

fn render_command_popup(frame: &mut Frame, area: Rect) {
    let block = pane_block("Command Popup", Borders::TOP | Borders::LEFT | Borders::RIGHT, false);
    let inner = block.inner(area);

    frame.render_widget(block, area);

    let [shortcuts_area, status_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(inner);

    frame.render_widget(
        Paragraph::new("f find market   1-6 focus pane   ↑↓ move   Enter open   w watch   ? help")
            .style(Style::default().fg(MUTED).bg(BG)),
        shortcuts_area,
    );
    frame.render_widget(
        Paragraph::new("Filter: politics volume>100k sort:move                               NORMAL")
            .style(Style::default().fg(MUTED).bg(BG)),
        status_area,
    );
}

fn render_pane(frame: &mut Frame, area: Rect, title: &str, borders: Borders, focused: bool) {
    frame.render_widget(pane_block(title, borders, focused), area);
}

fn pane_block(title: &str, borders: Borders, focused: bool) -> Block<'_> {
    let border_color = if focused { ACCENT } else { BORDER };

    Block::default()
        .borders(borders)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color).bg(BG))
        .style(Style::default().bg(BG))
        .title(title)
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