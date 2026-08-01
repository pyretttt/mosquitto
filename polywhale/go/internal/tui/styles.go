package tui

import (
	"strings"

	"github.com/charmbracelet/lipgloss"

	"polywhale/internal/polymarket"
)

// Color palette ported from the ratatui UI.
var (
	colorBG       = lipgloss.Color("0")
	colorBorder   = lipgloss.Color("#334155")
	colorAccent   = lipgloss.Color("10") // LightGreen
	colorPositive = lipgloss.Color("#22C55E")
	colorNegative = lipgloss.Color("#EF4444")
	colorWarning  = lipgloss.Color("#F59E0B")
	colorMuted    = lipgloss.Color("8") // DarkGray
	colorHeader   = lipgloss.Color("3") // Yellow
	colorWhite    = lipgloss.Color("15")
	colorGray     = lipgloss.Color("7")
	colorCyan     = lipgloss.Color("6")
	colorMagenta  = lipgloss.Color("5")
	colorRed      = lipgloss.Color("1")
	colorGreen    = lipgloss.Color("2")
	colorFocus    = lipgloss.Color("#6878F8")
)

// PuBu color scale used to animate the loading logo.
var pubuLogoColors = []lipgloss.Color{
	"#FFF7FB", "#F4EDF6", "#ECE7F2", "#DDDEEB", "#D0D1E6", "#BAC6DE",
	"#A6BDDB", "#8AB1D5", "#74A9CF", "#4D99C6", "#3690C0", "#147DB2",
	"#0570B0", "#045A8D", "#034872", "#023858",
}

func pubuLogoColor(index int) lipgloss.Color {
	mirrored := index % (len(pubuLogoColors) * 2)
	if mirrored >= len(pubuLogoColors) {
		mirrored = len(pubuLogoColors) - (mirrored % len(pubuLogoColors)) - 1
	}
	return pubuLogoColors[mirrored]
}

func fg(color lipgloss.Color) lipgloss.Style {
	return lipgloss.NewStyle().Foreground(color)
}

func fgBold(color lipgloss.Color) lipgloss.Style {
	return fg(color).Bold(true)
}

func activityKindColor(kind polymarket.ActivityKind) lipgloss.Color {
	switch kind {
	case polymarket.ActivityPositive:
		return colorPositive
	case polymarket.ActivityNegative:
		return colorNegative
	case polymarket.ActivityAccent:
		return colorAccent
	case polymarket.ActivityWarning:
		return colorWarning
	default:
		return colorMuted
	}
}

// truncPad clips or pads a plain (unstyled) string to exactly w cells.
func truncPad(s string, w int) string {
	if w <= 0 {
		return ""
	}
	width := lipgloss.Width(s)
	if width > w {
		runes := []rune(s)
		for len(runes) > 0 && lipgloss.Width(string(runes)) > w {
			runes = runes[:len(runes)-1]
		}
		s = string(runes)
		width = lipgloss.Width(s)
	}
	return s + strings.Repeat(" ", w-width)
}

// padLine pads a styled line to exactly w cells (clipping styled text is
// avoided by construction: callers pre-clip cell contents).
func padLine(line string, w int) string {
	width := lipgloss.Width(line)
	if width >= w {
		return line
	}
	return line + strings.Repeat(" ", w-width)
}

// pane draws a rounded-border box with an optional left-aligned title in the
// top border and an optional centered footer in the bottom border, matching
// the ratatui pane style.
func pane(title, footer string, width, height int, borderColor lipgloss.Color, contentLines []string) string {
	if width < 2 || height < 2 {
		return ""
	}
	innerW := width - 2
	border := fg(borderColor)

	var b strings.Builder
	b.WriteString(border.Render("╭" + embedInBorder(title, innerW, false) + "╮"))
	b.WriteString("\n")

	for i := 0; i < height-2; i++ {
		var line string
		if i < len(contentLines) {
			line = contentLines[i]
		}
		b.WriteString(border.Render("│"))
		b.WriteString(padLine(line, innerW))
		b.WriteString(border.Render("│"))
		b.WriteString("\n")
	}

	b.WriteString(border.Render("╰" + embedInBorder(footer, innerW, true) + "╯"))
	return b.String()
}

// embedInBorder places text inside a horizontal border run of length w.
func embedInBorder(text string, w int, centered bool) string {
	if text == "" || lipgloss.Width(text) > w {
		if lipgloss.Width(text) > w {
			text = truncPad(text, w)
			return text
		}
		return strings.Repeat("─", w)
	}
	remaining := w - lipgloss.Width(text)
	if centered {
		left := remaining / 2
		return strings.Repeat("─", left) + text + strings.Repeat("─", remaining-left)
	}
	return text + strings.Repeat("─", remaining)
}

// tableRow renders fixed-width cells joined with a single-space column gap.
// Each cell value is clipped to its width before styling.
type tableCell struct {
	text  string
	style lipgloss.Style
}

func tableRow(widths []int, cells []tableCell) string {
	parts := make([]string, 0, len(cells))
	for i, cell := range cells {
		w := 0
		if i < len(widths) {
			w = widths[i]
		}
		parts = append(parts, cell.style.Render(truncPad(cell.text, w)))
	}
	return strings.Join(parts, " ")
}

// overlayBottom replaces the bottom lines of a base view with a popup,
// emulating ratatui's Clear + render at a bottom-anchored Rect.
func overlayBottom(base, popup string, width, height int) string {
	baseLines := strings.Split(base, "\n")
	for len(baseLines) < height {
		baseLines = append(baseLines, "")
	}
	baseLines = baseLines[:height]

	popupLines := strings.Split(popup, "\n")
	start := height - len(popupLines)
	if start < 0 {
		popupLines = popupLines[-start:]
		start = 0
	}
	for i, line := range popupLines {
		baseLines[start+i] = padLine(line, width)
	}
	return strings.Join(baseLines, "\n")
}

func centerLine(line string, width int) string {
	lineWidth := lipgloss.Width(line)
	if lineWidth >= width {
		return line
	}
	return strings.Repeat(" ", (width-lineWidth)/2) + line
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
