package tui

import (
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"polywhale/internal/logging"
)

// logModel is the Go stand-in for the tui-logger overlay: it shows the
// in-memory log buffer with scrolling and an app-target filter.
type logModel struct {
	memLog        *logging.Memory
	scrollFromEnd int // 0 = follow the tail
	appFilter     bool
	help          [3]string
}

func newLogModel(memLog *logging.Memory) logModel {
	return logModel{
		memLog: memLog,
		help: [3]string{
			"Q: Quit | j/k: Scroll page | ↑/↓: Scroll line | s: Switch app logs filter",
			"Esc: Cancel scroll",
			"",
		},
	}
}

// handleKey returns (handled, closed).
func (l *logModel) handleKey(msg tea.KeyMsg) (bool, bool) {
	switch msg.String() {
	case "q":
		return true, true
	case "esc":
		l.scrollFromEnd = 0
		return true, false
	case "k":
		l.scrollFromEnd += 10
		return true, false
	case "j":
		l.scrollFromEnd = max(l.scrollFromEnd-10, 0)
		return true, false
	case "up":
		l.scrollFromEnd++
		return true, false
	case "down":
		l.scrollFromEnd = max(l.scrollFromEnd-1, 0)
		return true, false
	case "s":
		l.appFilter = !l.appFilter
		l.scrollFromEnd = 0
		return true, false
	case "tab", " ", "f", "h", "+", "-", "left", "right":
		// Accepted (and ignored) for parity with the tui-logger keymap.
		return true, false
	}
	return false, false
}

func (l *logModel) visibleLines() []string {
	lines := l.memLog.Lines()
	if !l.appFilter {
		return lines
	}
	filtered := make([]string, 0, len(lines))
	for _, line := range lines {
		if strings.Contains(line, ":app:") {
			filtered = append(filtered, line)
		}
	}
	return filtered
}

func (l *logModel) view(width, height int) string {
	const helpH = 4
	logH := max(height-helpH, 0)

	lines := l.visibleLines()
	end := max(len(lines)-l.scrollFromEnd, 0)
	start := max(end-logH, 0)
	window := lines[start:end]

	var b strings.Builder
	for i := 0; i < logH; i++ {
		if i < len(window) {
			b.WriteString(padLine(styleLogLine(truncPad(window[i], width)), width))
		}
		b.WriteString("\n")
	}

	gray := fg(colorGray)
	b.WriteString(centerLine(gray.Render(l.help[0]), width) + "\n")
	b.WriteString(centerLine(gray.Render(l.help[1]), width) + "\n")
	b.WriteString(centerLine(gray.Render(l.help[2]), width))
	return b.String()
}

// styleLogLine colors a line by its abbreviated level (TIME:LEVEL:TARGET:msg).
func styleLogLine(line string) string {
	parts := strings.SplitN(line, ":", 5)
	if len(parts) < 4 {
		return line
	}
	level := parts[3]
	switch level {
	case "E":
		return fg(colorRed).Render(line)
	case "W":
		return fg(colorHeader).Render(line)
	case "I":
		return fg(colorCyan).Render(line)
	case "D":
		return fg(colorGreen).Render(line)
	case "T":
		return fg(colorMagenta).Render(line)
	}
	return line
}
