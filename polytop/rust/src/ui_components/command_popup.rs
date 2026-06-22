use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect, Margin},
    style::{Color, Modifier, Style},
    text::{Span, Text, Line},
    widgets::{Block, Cell, Clear, Row, Table, Paragraph, Borders},
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
        Table::new(
            command_rows(command_pallette),
            [Constraint::Length(12), Constraint::Fill(1)],
        )
        .column_spacing(2)
        .style(Style::default().fg(Color::Gray)),
        commands_area.inner(Margin {
            horizontal: 2,
            vertical: 1,
        }),
    );

    frame.render_widget(
        Paragraph::new(command_pallette.input_text_ui())
        .block(
            Block::default()
            .borders(Borders::TOP)
                .border_style(Style::default().fg(Color::Gray)),
        ),
        input_area.inner(Margin {
            horizontal: 1,
            vertical: 0,
        }),
    );
}

fn command_rows(command_pallette: &CommandPallette) -> impl Iterator<Item = Row<'static>> + use<'_> {
    let mut commands = command_pallette.available_commands().enumerate().peekable();

    let placeholder = if commands.peek().is_none() {
        Some(Row::new([Cell::from(Span::styled(
            "No commands",
            Style::default().fg(Color::DarkGray),
        ))]))
    } else {
        None
    };

    placeholder.into_iter().chain(commands.map(|(index, command)| {
        let (name_style, description_style) = if index == 0 {
            (
                Style::default()
                    .fg(FOCUS_COLOR)
                    .add_modifier(Modifier::BOLD),
                Style::default().fg(FOCUS_COLOR),
            )
        } else {
            (
                Style::default().fg(Color::White),
                Style::default().fg(Color::DarkGray),
            )
        };

        Row::new([
            Cell::from(Span::styled(command.name().to_owned(), name_style)),
            Cell::from(Span::styled(
                command.description().to_owned(),
                description_style,
            )),
        ])
    }))
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

impl CommandPallette {
    fn input_text_ui(&self) -> Text<'_> {
        let line = if self.input_text.is_empty() {
            let span = Span::styled(self.input_placeholder, Style::default().fg(Color::DarkGray));
            Line::from_iter([span].into_iter())
        } else {
            let span = Span::styled(self.input_text.as_str(), Style::default().fg(Color::White));
            if let Some(command) = self.command_to_complete() {
                let suffix = &command.name()[self.input_text.len()..];
                let cmd_span = Span::styled(suffix, Style::default().fg(Color::DarkGray));
                Line::from_iter([span, cmd_span].into_iter())
            } else {
                Line::from_iter([span].into_iter())
            }
        };


        Text::from(line).left_aligned()
    }
}