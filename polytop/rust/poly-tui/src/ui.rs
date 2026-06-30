use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Layout, Offset},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Gauge, Paragraph},
};
use ratatui::DefaultTerminal;
use throbber_widgets_tui::{Throbber, BRAILLE_EIGHT, WhichUse};

use crate::models::app_state::{
    AppState,
    Page,
    app_reducer,
    IntroPage,
};
use crate::models::loading_page::LoadingPage;
use crate::env::Env;
use crate::ui_components::command_popup::draw_command_popup;
use crate::ui_components::top_page_ui::top_page_ui;

const POLYTOP_LOGO: [&str; 6] = [
    "██████╗  ██████╗ ██╗  ██╗   ██╗████████╗ ██████╗ ██████╗ ",
    "██╔══██╗██╔═══██╗██║  ╚██╗ ██╔╝╚══██╔══╝██╔═══██╗██╔══██╗",
    "██████╔╝██║   ██║██║   ╚████╔╝    ██║   ██║   ██║██████╔╝",
    "██╔═══╝ ██║   ██║██║    ╚██╔╝     ██║   ██║   ██║██╔═══╝ ",
    "██║     ╚██████╔╝███████╗██║      ██║   ╚██████╔╝██║     ",
    "╚═╝      ╚═════╝ ╚══════╝╚═╝      ╚═╝    ╚═════╝ ╚═╝     ",
];

const PUBU_LOGO_COLORS: [Color; 16] = [
    Color::Rgb(255, 247, 251),
    Color::Rgb(244, 237, 246),
    Color::Rgb(236, 231, 242),
    Color::Rgb(221, 222, 235),
    Color::Rgb(208, 209, 230),
    Color::Rgb(186, 198, 222),
    Color::Rgb(166, 189, 219),
    Color::Rgb(138, 177, 213),
    Color::Rgb(116, 169, 207),
    Color::Rgb(77, 153, 198),
    Color::Rgb(54, 144, 192),
    Color::Rgb(20, 125, 178),
    Color::Rgb(5, 112, 176),
    Color::Rgb(4, 90, 141),
    Color::Rgb(3, 72, 114),
    Color::Rgb(2, 56, 88),
];

fn draw_app(frame: &mut Frame, state: &AppState, env: &Env) {
    frame.render_widget(Block::new().style(Style::default().bg(Color::Black)), frame.area());
    match &state.page {
        Page::Intro(intro) => draw_intro_page(frame, state, intro),
        Page::Top(top) => top_page_ui(frame, state, top, env),
        Page::LoadingPage(loading) => draw_loading_page(frame, loading),
        Page::Help(_) => (),
    }

    if let Some(command_pallette) = &state.command_pallette {
        draw_command_popup(frame, command_pallette);
    }
}

fn pubu_logo_color(logo_color_index: usize) -> Color {
    let mut mirrored_index = logo_color_index % (PUBU_LOGO_COLORS.len() * 2);
    if mirrored_index >= PUBU_LOGO_COLORS.len() {
        mirrored_index = PUBU_LOGO_COLORS.len() - (mirrored_index % PUBU_LOGO_COLORS.len()) - 1;
    }
    PUBU_LOGO_COLORS[mirrored_index]
}

fn draw_intro_page(frame: &mut Frame, state: &AppState, intro: &IntroPage) {
    let [title_area, text_area, counter_area, help_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Fill(1),
    ])
    .margin(1)
    .areas(frame.area());

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "intro.title.as_str()",
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ))),
        title_area,
    );

    frame.render_widget(Paragraph::new("intro.text.as_str()"), text_area);

    frame.render_widget(
        Paragraph::new(format!("counter: {}", 0)),
        counter_area,
    );

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Press q to quit",
            Style::default().fg(Color::DarkGray),
        ))),
        help_area,
    );
}

pub fn draw_loading_page(frame: &mut Frame, loading: &LoadingPage) {
    let [logo_area, throbber_area, fill_area, gauge_area, tip_area] = Layout::vertical([
        Constraint::Length(POLYTOP_LOGO.len() as u16),
        Constraint::Length(3),
        Constraint::Fill(1),
        Constraint::Length(3),
        Constraint::Length(2),
    ])
    .margin(3)
    .areas(frame.area());

    frame.render_widget(
        Paragraph::new(POLYTOP_LOGO.join("\n"))
            .alignment(Alignment::Center)
            .style(
                Style::default()
                    .fg(
                        pubu_logo_color(loading.logo_color_index)
                    )
                    .add_modifier(Modifier::BOLD),
            ),
        logo_area,
    );

    let throbber = Throbber::default()
        .throbber_set(BRAILLE_EIGHT)
        .use_type(WhichUse::Spin)
        .throbber_style(Style::default().fg(Color::Cyan))
        .label(Span::styled(
            &loading.throbbler_caption,
            Style::default().fg(Color::White),
        ));
    frame.render_widget(
        Paragraph::new(throbber.to_line(&loading.throbbler_state)).alignment(Alignment::Center),
        throbber_area.offset(Offset::new(0, 1)),
    );

    frame.render_widget(Block::new().style(Style::default().fg(Color::Black)), fill_area);

    let progress = loading.progress.clamp(0.0, 1.0) as f64;
    frame.render_widget(
        Gauge::default()
            .block(Block::bordered().title("Progress"))
            .gauge_style(Style::default().fg(Color::Cyan))
            .ratio(progress)
            .label(format!("{:.0}%", progress * 100.0)),
        gauge_area,
    );

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("Tip: ", Style::default().fg(Color::DarkGray)),
            Span::raw(loading.loading_tip),
        ]))
        .alignment(Alignment::Center),
        tip_area.offset(Offset::new(0, 1)),
    );
}

pub async fn run(
    app_state: &mut AppState,
    env: &mut Env,
    terminal: &mut DefaultTerminal,
) -> color_eyre::Result<()> {
    while app_state.running {
        terminal.draw(|frame| draw_app(frame, &mut *app_state, env))?;
        let mut event = env.receiver
            .recv()
            .await
            .ok_or(color_eyre::eyre::eyre!("Event receiver closed"))?;
        app_reducer(app_state, &mut event, env);
        if !app_state.running {
            break;
        }
    }
    Ok(())
}
