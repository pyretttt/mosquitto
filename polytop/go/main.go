package main

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/polytop/go/model"
)

type AppModel model.AppModel

func (m AppModel) Init() tea.Cmd {
	return nil
}

func (m AppModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			return m, tea.Quit
		}
	}
	return m, nil
}

func (m AppModel) View() string {
	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#7D56F4")).
		Render("Polytop")

	body := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#FAFAFA")).
		Render("Hello, world!")

	help := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#626262")).
		Render("Press q to quit")

	return lipgloss.JoinVertical(lipgloss.Left, title, "", body, "", help)
}

func main() {
	p := tea.NewProgram(AppModel{}, tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
