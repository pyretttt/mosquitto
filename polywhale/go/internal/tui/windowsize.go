package tui

import (
	"fmt"
	"strings"
)

// windowSizeModel is the "terminal too small" full-screen overlay.
type windowSizeModel struct {
	currentWidth   int
	currentHeight  int
	requiredWidth  int
	requiredHeight int
}

func newWindowSizeModel(requiredWidth, requiredHeight int) windowSizeModel {
	return windowSizeModel{
		requiredWidth:  requiredWidth,
		requiredHeight: requiredHeight,
	}
}

func (w *windowSizeModel) view(width, height int) string {
	current := fmt.Sprintf("%dx%d", w.currentWidth, w.currentHeight)
	required := fmt.Sprintf("%dx%d", w.requiredWidth, w.requiredHeight)

	content := []string{
		fgBold(colorHeader).Render("Terminal too small"),
		"",
		fg(colorMuted).Render("Current:  ") + fg(colorRed).Render(current),
		fg(colorMuted).Render("Required: ") + fg(colorGreen).Render(required),
		fg(colorMuted).Render("Resize your terminal to continue"),
	}

	topPad := max((height-len(content))/2, 0)
	var b strings.Builder
	for i := 0; i < topPad; i++ {
		b.WriteString("\n")
	}
	for i, line := range content {
		b.WriteString(centerLine(line, width))
		if topPad+i < height-1 {
			b.WriteString("\n")
		}
	}
	return b.String()
}
