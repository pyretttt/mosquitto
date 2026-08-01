package tui

import (
	"strings"
	"testing"

	"polywhale/internal/polymarket"
)

func TestLoadingViewRenders(t *testing.T) {
	loading := newLoadingModel()
	loading.progress = 0.5
	view := loading.view(120, 40)

	for _, want := range []string{"██████╗", "Cooking smooth performance...", "Progress", "50%", "Tip: "} {
		if !strings.Contains(view, want) {
			t.Errorf("loading view missing %q", want)
		}
	}
}

func TestTopViewRendersMockPanes(t *testing.T) {
	top := newTopModel(120, 40)
	markets := []polymarket.Market{
		polymarket.NewMarket("Will it happen?", "will-it", false, 0.63, 0.37, 12000, 0.02, 0.01, polymarket.UmaPending),
	}
	event := polymarket.NewEvent("1", "Test event", "test-event", false, 42000, markets)
	event.SetRank(1)
	top.events.data = polymarket.EventsData{Events: []polymarket.Event{event}, NextCursor: 30}
	top.events.refreshLabels()

	view := top.view()
	for _, want := range []string{
		" POLYWHALE ",
		"[1] - Top Events: all",
		"Test event",
		"Will it happen?",
		"[2] - Selected Market: summary",
		"[3] - Chart + Activity",
		"Keybindings:",
		"net online",
	} {
		if !strings.Contains(view, want) {
			t.Errorf("top view missing %q", want)
		}
	}

	lines := strings.Split(view, "\n")
	if len(lines) != 40 {
		t.Errorf("top view should be 40 lines, got %d", len(lines))
	}
}

func TestWindowSizeOverlay(t *testing.T) {
	overlay := newWindowSizeModel(120, 40)
	overlay.currentWidth, overlay.currentHeight = 80, 24
	view := overlay.view(80, 24)

	for _, want := range []string{"Terminal too small", "80x24", "120x40", "Resize your terminal to continue"} {
		if !strings.Contains(view, want) {
			t.Errorf("window size view missing %q", want)
		}
	}
}

func TestPaletteFilterAndComplete(t *testing.T) {
	palette := newPaletteModel()
	palette.changeText("lo")

	available := palette.availableCommands()
	if len(available) != 1 || available[0] != commandLog {
		t.Fatalf("expected only the log command, got %v", available)
	}
	if cmd, ok := palette.commandToComplete(); !ok || cmd != commandLog {
		t.Fatalf("expected log completion")
	}
	if view := palette.view(120, 40); !strings.Contains(view, "Show logs") {
		t.Errorf("palette view missing filtered command description")
	}
}
