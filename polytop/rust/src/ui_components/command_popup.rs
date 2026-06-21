use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Clear, Paragraph, Wrap},
};

use crate::models::command::CommandPallette;

const FOCUS_COLOR: Color = Color::Rgb(0x68, 0x78, 0xF8);

pub fn draw_command_popup(frame: &mut Frame, command_pallette: &CommandPallette) {
    let area = popup_area(frame.area());
    let [commands_area, input_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(3)]).areas(area);

    frame.render_widget(Clear, area);
    frame.render_widget(
        Block::bordered()
            .title(" Commands ")
            .style(Style::default().fg(Color::White)),
        area,
    );

    frame.render_widget(
        Paragraph::new(command_lines(command_pallette))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(Color::Gray)),
        commands_area.inner(ratatui::layout::Margin {
            horizontal: 2,
            vertical: 1,
        }),
    );

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("/", Style::default().fg(Color::DarkGray)),
            Span::raw(command_pallette.input_text.as_str()),
        ]))
        .block(Block::bordered().title(" Command "))
        .style(Style::default().fg(Color::White)),
        input_area,
    );
}

fn command_lines(command_pallette: &CommandPallette) -> impl Iterator<Item = Line<'static>> {
    let commands = command_pallette.available_commands();
    if commands.len() {
        return Some(Line::from(Span::styled(
            "No commands",
            Style::default().fg(Color::DarkGray),
        )));
    }

    commands
        .iter()
        .enumerate()
        .flat_map(|(index, command)| {
            let style = if index == 0 {
                Style::default()
                    .fg(FOCUS_COLOR)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };

            [
                Line::from(Span::styled(command.name().to_owned(), style)),
                Line::from(Span::styled(
                    command.description().to_owned(),
                    if index == 0 {
                        Style::default().fg(FOCUS_COLOR)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    },
                )),
                Line::default(),
            ]
        })
}

fn popup_area(area: Rect) -> Rect {
    let height = area.height.saturating_mul(2).saturating_div(3).clamp(8, 18);
    let height = height.min(area.height);

    Rect {
        x: area.x,
        y: area.y + area.height.saturating_sub(height),
        width: area.width,
        height,
    }
}
