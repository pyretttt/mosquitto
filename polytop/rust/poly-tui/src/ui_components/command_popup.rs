use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect, Margin},
    style::{Color, Modifier, Style},
    text::{Span, Text, Line},
    widgets::{Block, Cell, Clear, Row, Table, Paragraph, Borders, BorderType},
};

use crate::features::command::{Command, CommandPallette, TextAreaState};

const FOCUS_COLOR: Color = Color::Rgb(0x68, 0x78, 0xF8);


pub fn draw_command_popup(frame: &mut Frame, command_pallette: &CommandPallette) {
    let area = popup_area(frame.area());
    let [commands_area, input_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(3)]).areas(area);

    frame.render_widget(Clear, area);
    frame.render_widget(Block::new().style(Style::default().bg(Color::Black)), area);
    frame.render_widget(
        Block::bordered()
            .title(" Commands ")
            .border_type(BorderType::Rounded)
            .style(Style::default().fg(Color::White)),
        area,
    );

    let command_rows = command_rows(command_pallette.available_commands());
    let table = Table::new(
        command_rows,
        [Constraint::Length(12), Constraint::Fill(1)],
    )
    .column_spacing(2)
    .style(Style::default().fg(Color::Gray))
    .row_highlight_style(
        Style::default()
            .fg(FOCUS_COLOR)
            .add_modifier(Modifier::BOLD),
    );

    frame.render_stateful_widget(
        table,
        commands_area.inner(Margin {
            horizontal: 2,
            vertical: 1,
        }),
        &mut command_pallette.table_state.clone(),
    );

    frame.render_widget(
        Paragraph::new(
            command_pallette.text_area_state.ui(
                command_pallette
                    .command_to_complete()
                    .map(|command| command.name())
            )
        )
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

fn command_rows<'a>(commands: impl Iterator<Item = &'a Command>) -> impl Iterator<Item = Row<'a>> {
    let mut commands_iter = commands.peekable();

    let fallback = if commands_iter.peek().is_none() {
        Some(Row::new([Cell::from(Span::styled(
            "No commands",
            Style::default().fg(Color::DarkGray),
        ))]))
    } else {
        None
    };

    commands_iter
        .map(|command| {
            Row::new([
                Cell::from(Span::styled(
                    command.name(),
                    Style::default().fg(Color::White),
                )),
                Cell::from(Span::styled(
                    command.description(),
                    Style::default().fg(Color::DarkGray),
                )),
            ])
        })
        .chain(fallback) // Option<Row> implements IntoIterator
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

impl TextAreaState {
    fn ui<'a>(&'a self, text_to_complete: Option<&'a str>) -> Text<'a> {
        let line = if self.text.is_empty() {
            let span = Span::styled(self.input_placeholder, Style::default().fg(Color::DarkGray));
            Line::from_iter([span].into_iter())
        } else {
            let span = Span::styled(self.text.as_str(), Style::default().fg(Color::White));
            if let Some(text_to_complete) = text_to_complete {
                let suffix = &text_to_complete[self.text.len()..];
                let cmd_span = Span::styled(suffix, Style::default().fg(Color::DarkGray));
                Line::from_iter([span, cmd_span].into_iter())
            } else {
                Line::from_iter([span].into_iter())
            }
        };

        Text::from(line).left_aligned()
    }
}