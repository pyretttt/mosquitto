package main

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"

	"polywhale/internal/config"
	"polywhale/internal/logging"
	"polywhale/internal/polymarket"
	"polywhale/internal/tui"
)

func main() {
	memLog, _, err := logging.Setup()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to set up logging: %v\n", err)
		os.Exit(1)
	}

	model := tui.NewModel(config.Get(), polymarket.NewClient(), memLog)
	program := tea.NewProgram(model, tea.WithAltScreen())
	if _, err := program.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "polywhale exited with error: %v\n", err)
		os.Exit(1)
	}
}
