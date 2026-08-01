package tui

import (
	"fmt"
	"math/rand/v2"
	"strings"
)

var loadingTips = []string{
	"Press `q` to quit",
	"Press `/` to open command palette",
	"Press `?` to open help",
	"Press `Ctrl+c` to quit",
}

const (
	maxFakeProgress   = 0.87
	throbberCaption   = "Cooking smooth performance..."
	loadingPageMargin = 3
)

// BRAILLE_EIGHT spinner frames from throbber-widgets-tui.
var brailleEightFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

var whaleArt = []string{
	"     .-'               ",
	"'--./ /     _.---.     ",
	"'-,  (__..-`       \\  ",
	"   \\          .     | ",
	"    `,.__.   ,__.--/   ",
	"      '._/_.'___.-`    ",
}

var polywhaleLogo = []string{
	"██████╗  ██████╗ ██╗  ██╗   ██╗██╗    ██╗██╗  ██╗ █████╗ ██╗     ███████╗",
	"██╔══██╗██╔═══██╗██║  ╚██╗ ██╔╝██║    ██║██║  ██║██╔══██╗██║     ██╔════╝",
	"██████╔╝██║   ██║██║   ╚████╔╝ ██║ █╗ ██║███████║███████║██║     █████╗  ",
	"██╔═══╝ ██║   ██║██║    ╚██╔╝  ██║███╗██║██╔══██║██╔══██║██║     ██╔══╝  ",
	"██║     ╚██████╔╝███████╗██║   ╚███╔███╔╝██║  ██║██║  ██║███████╗███████╗",
	"╚═╝      ╚═════╝ ╚══════╝╚═╝    ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝",
}

type loadingModel struct {
	progress       float64
	tip            string
	tipIndex       int
	caption        string
	logoColorIndex int
	spinnerFrame   int
	tickCount      uint16
	finished       bool
}

func newLoadingModel() loadingModel {
	return loadingModel{
		tip:     loadingTips[0],
		caption: throbberCaption,
	}
}

func (l *loadingModel) tick(tickRate float64) {
	l.tickCount++
	if l.tickCount%3 == 0 {
		l.spinnerFrame++
		l.logoColorIndex++
	}
	if tipEvery := uint16(tickRate * 2); tipEvery > 0 && l.tickCount%tipEvery == 0 {
		l.tipIndex = (l.tipIndex + 1) % len(loadingTips)
		l.tip = loadingTips[l.tipIndex]
	}
	if !l.finished && l.tickCount%10 == 0 {
		l.progress = min(l.progress+0.01+rand.Float64()*0.09, maxFakeProgress)
	}
}

func (l *loadingModel) view(width, height int) string {
	innerW := width - loadingPageMargin*2
	innerH := height - loadingPageMargin*2
	if innerW < 10 || innerH < 5 {
		return ""
	}

	logoStyle := fgBold(pubuLogoColor(l.logoColorIndex))
	var lines []string
	for _, artLine := range whaleArt {
		lines = append(lines, centerLine(logoStyle.Render(artLine), innerW))
	}
	lines = append(lines, "")
	for _, logoLine := range polywhaleLogo {
		lines = append(lines, centerLine(logoStyle.Render(logoLine), innerW))
	}

	// Throbber (3 rows tall, drawn with a 1-row offset).
	frame := brailleEightFrames[l.spinnerFrame%len(brailleEightFrames)]
	throbberLine := fg(colorCyan).Render(frame) + " " + fg(colorWhite).Render(l.caption)
	lines = append(lines, "", centerLine(throbberLine, innerW), "")

	// Fill area pushes the gauge + tip to the bottom.
	used := len(lines) + 3 + 2 // gauge is 3 rows, tip block is 2
	for i := 0; i < innerH-used; i++ {
		lines = append(lines, "")
	}

	lines = append(lines, l.gaugeLines(innerW)...)
	lines = append(lines, "", centerLine(fg(colorMuted).Render("Tip: ")+l.tip, innerW))

	var b strings.Builder
	for i := 0; i < loadingPageMargin; i++ {
		b.WriteString("\n")
	}
	margin := strings.Repeat(" ", loadingPageMargin)
	for i, line := range lines {
		if i >= innerH {
			break
		}
		b.WriteString(margin + line + "\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

// gaugeLines renders a bordered progress gauge like ratatui's Gauge widget.
func (l *loadingModel) gaugeLines(width int) []string {
	progress := clampFloat(l.progress, 0, 1)
	innerW := width - 2
	filled := int(progress * float64(innerW))
	label := fmt.Sprintf("%.0f%%", progress*100)

	bar := make([]rune, innerW)
	for i := range bar {
		if i < filled {
			bar[i] = '█'
		} else {
			bar[i] = ' '
		}
	}
	labelStart := (innerW - len(label)) / 2
	var left, mid, right string
	if labelStart >= 0 {
		left = string(bar[:labelStart])
		right = string(bar[labelStart+len(label):])
		mid = label
	} else {
		left = string(bar)
	}

	barLine := fg(colorCyan).Render(left) + fg(colorWhite).Render(mid) + fg(colorCyan).Render(right)
	border := fg(colorWhite)
	return []string{
		border.Render("┌" + embedInBorderPlain("Progress", width-2) + "┐"),
		border.Render("│") + barLine + border.Render("│"),
		border.Render("└" + strings.Repeat("─", width-2) + "┘"),
	}
}

func embedInBorderPlain(text string, w int) string {
	if len(text) > w {
		return strings.Repeat("─", w)
	}
	return text + strings.Repeat("─", w-len(text))
}

func clampFloat(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
