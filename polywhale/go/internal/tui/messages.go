package tui

import (
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"polywhale/internal/polymarket"
)

// tickMsg drives animations at TICK_RATE ticks per second.
type tickMsg time.Time

// debounceElapsedMsg fires one second after Enter is pressed; the token
// guards against stale presses (the "debounce example" in the Rust app).
type debounceElapsedMsg struct{ token string }

// eventsLoadedMsg / eventsLoadFailedMsg complete an events page request.
type eventsLoadedMsg struct {
	data    polymarket.EventsData
	session string
}

type eventsLoadFailedMsg struct{ session string }

// hideErrorMsg clears the transient error banner if its token still matches.
type hideErrorMsg struct{ token string }

func tickCmd(interval time.Duration) tea.Cmd {
	return tea.Tick(interval, func(t time.Time) tea.Msg { return tickMsg(t) })
}
