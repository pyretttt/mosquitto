package tui

import (
	"strings"

	"github.com/charmbracelet/lipgloss"

	"polywhale/internal/polymarket"
)

const (
	barIntermediate = "┃"
	barTop          = "┳"
	barBottom       = "┻"
)

func (t *topModel) view() string {
	width, height := t.width, t.height
	if width < 4 || height < 8 {
		return ""
	}

	var b strings.Builder
	// Outer block: top border only, with the app title embedded.
	b.WriteString(fg(colorBorder).Render(embedInBorder(t.leftTitle, width, false)))
	b.WriteString("\n")

	dashboardH := height - 1
	searchFocused := t.events.search != nil && t.events.search.focused
	searchH := 0
	if searchFocused {
		searchH = 3
	}
	const statusH, cmdH = 1, 3

	budget := dashboardH - statusH - searchH - cmdH
	eventsH := clampInt(t.tableHeight(), 0, max(budget, 0))
	lowerH := max(budget-eventsH, 0)

	b.WriteString(t.renderStatusBar(width))
	b.WriteString("\n")
	if searchFocused {
		b.WriteString(t.renderSearchInput(width))
		b.WriteString("\n")
	}
	if eventsH > 0 {
		b.WriteString(t.renderTopEvents(width, eventsH))
		b.WriteString("\n")
	}
	if lowerH > 0 {
		b.WriteString(t.renderLowerPanes(width, lowerH))
		b.WriteString("\n")
	}
	b.WriteString(t.renderKeyBindings(width))
	return b.String()
}

func (t *topModel) renderStatusBar(width int) string {
	status := &t.status

	onlineIcon, onlineColor, onlineLabel := " ✓ ", colorPositive, "net online"
	if !status.isOnline {
		onlineIcon, onlineColor, onlineLabel = " ✗ ", colorNegative, "net offline"
	}
	wsLabel, wsColor := "   ws live", colorAccent
	if !status.wsLive {
		wsLabel, wsColor = "   ws down", colorNegative
	}

	var left string
	if t.errorMsg != nil {
		left = fg(colorWarning).Render("⚠️ Error: ") + fg(colorNegative).Render(t.errorMsg.text)
	}

	right := fg(onlineColor).Render(onlineIcon) +
		fg(colorGray).Render(onlineLabel) +
		fg(wsColor).Render(wsLabel) +
		fg(colorWhite).Render(status.latencyLabel) +
		fg(colorGray).Render(status.refreshLabel) +
		fg(colorGray).Render(status.modeLabel)

	gap := width - 2 - lipgloss.Width(left) - lipgloss.Width(right)
	if gap < 1 {
		gap = 1
	}
	return " " + left + strings.Repeat(" ", gap) + right + " "
}

func (t *topModel) renderSearchInput(width int) string {
	search := t.events.search
	var content string
	if search.term == "" {
		content = fg(colorMuted).Render(search.placeholder)
	} else {
		content = fg(colorWhite).Render(search.term)
	}
	return pane(
		"Press enter to apply search: ",
		"Tab to switch search type - "+search.searchType.String(),
		width, 3, colorWarning,
		[]string{" " + content},
	)
}

// eventsColumnWidths ports events_column_constraints: fixed widths with the
// event title column absorbing the remaining space (Min(28)).
func eventsColumnWidths(innerW int) []int {
	fixed := 3 + 3 + 7*5 // all non-title columns
	spacing := 7         // 7 single-space gaps between 8 columns
	return []int{3, 3, max(innerW-fixed-spacing, 28), 7, 7, 7, 7, 7}
}

func (t *topModel) renderTopEvents(width, height int) string {
	borderColor := colorBorder
	if t.currentPane == 1 {
		borderColor = colorAccent
	}
	innerW := width - 2
	widths := eventsColumnWidths(innerW)

	events := t.events.eventSlice()
	selected := t.events.selected
	if len(events) > 0 {
		selected = min(selected, len(events)-1)
	}

	lines := []string{t.eventsHeaderRow(widths)}
	for index, event := range events {
		isSelected := index == selected
		lines = append(lines, eventRow(&event, isSelected, widths))
		if isSelected {
			lines = append(lines, t.marketRows(&event, widths)...)
		}
	}

	return pane(t.events.titleLabel, t.events.footerLabel, width, height, borderColor, lines)
}

func (t *topModel) eventsHeaderRow(widths []int) string {
	header := fgBold(colorHeader)
	cells := make([]tableCell, 0, 8)
	for _, label := range []string{"#", "★", "Event", "Yes", "No", "24h", "Move", "Spread"} {
		cells = append(cells, tableCell{text: label, style: header})
	}
	return tableRow(widths, cells)
}

func eventRow(event *polymarket.Event, isSelected bool, widths []int) string {
	rankColor, titleColor := colorMuted, colorWhite
	if isSelected {
		rankColor, titleColor = colorWarning, colorWarning
	}
	return tableRow(widths, []tableCell{
		{text: event.RankLabel, style: fg(rankColor)},
		{text: event.BookmarkLabel, style: fg(colorWarning)},
		{text: event.Title, style: fg(titleColor)},
		{text: "—", style: fg(colorMuted)},
		{text: "—", style: fg(colorMuted)},
		{text: event.VolumeLabel, style: fg(colorWhite)},
		{text: event.MarketsCountLabel, style: fg(colorMuted)},
		{text: "", style: fg(colorMuted)},
	})
}

// marketRows renders the nested markets table for the selected event,
// keeping the selection visible within maxMarketsTableHeight rows.
func (t *topModel) marketRows(event *polymarket.Event, widths []int) []string {
	total := len(event.Markets)
	if total == 0 {
		return nil
	}
	visible := min(total, maxMarketsTableHeight)
	offset := 0
	if t.events.marketsFocused {
		offset = clampInt(t.events.marketsSelected-(visible-1), 0, total-visible)
	}

	rows := make([]string, 0, visible)
	for i := offset; i < offset+visible; i++ {
		bar := barIntermediate
		switch {
		case total == 1:
			bar = barIntermediate
		case i == 0:
			bar = barTop
		case i == total-1:
			bar = barBottom
		}
		highlighted := t.events.marketsFocused && i == t.events.marketsSelected
		rows = append(rows, marketRow(&event.Markets[i], bar, highlighted, widths))
	}
	return rows
}

func marketRow(market *polymarket.Market, bar string, highlighted bool, widths []int) string {
	isActiveMarket := market.ResolutionStatus == polymarket.UmaPending ||
		market.ResolutionStatus == polymarket.UmaUnknown
	titleColor := colorGray
	if !isActiveMarket {
		titleColor = colorMuted
	}

	styleOr := func(base lipgloss.Style) lipgloss.Style {
		if highlighted {
			return fgBold(colorWarning)
		}
		return base
	}

	return tableRow(widths, []tableCell{
		{text: "", style: fg(colorMuted)},
		{text: market.BookmarkLabel, style: styleOr(fg(colorWarning))},
		{text: bar + "  " + market.Title, style: styleOr(fg(titleColor))},
		{text: market.YesLabel, style: styleOr(fg(colorPositive))},
		{text: market.NoLabel, style: styleOr(fg(colorNegative))},
		{text: market.VolumeLabel, style: styleOr(fg(colorWhite))},
		{text: market.MovementLabel, style: styleOr(fg(activityKindColor(market.MovementKind)))},
		{text: market.SpreadLabel, style: styleOr(fg(colorMuted))},
	})
}

func (t *topModel) renderLowerPanes(width, height int) string {
	leftW := width / 2
	rightW := width - leftW
	left := t.renderSelectedMarket(leftW, height)
	right := t.renderChartActivity(rightW, height)
	return lipgloss.JoinHorizontal(lipgloss.Top, left, right)
}

func (t *topModel) renderSelectedMarket(width, height int) string {
	market := &t.selectedMarket.selectedMarket
	borderColor := colorBorder
	if t.currentPane == 2 {
		borderColor = colorAccent
	}

	metricLine := func(label, value string) string {
		return " " + fg(colorMuted).Render(label) + fg(colorWhite).Render(value)
	}

	lines := []string{
		" " + fg(colorWhite).Render(t.selectedMarket.title),
		"",
		" " + fg(colorPositive).Render("yes  ") + fgBold(colorPositive).Render(market.YesLabel) +
			fg(colorMuted).Render(market.YesQuotesLabel),
		" " + fg(colorNegative).Render("no   ") + fgBold(colorNegative).Render(market.NoLabel) +
			fg(colorMuted).Render(market.NoQuotesLabel),
		"",
		metricLine("volume 24h    ", market.VolumeLabel),
		metricLine("liquidity     ", market.LiquidityLabel),
		metricLine("open interest ", market.OpenInterestLabel),
		metricLine("end date      ", market.EndDate),
	}
	return pane(" [2] - Selected Market: summary ", "", width, height, borderColor, lines)
}

func (t *topModel) renderChartActivity(width, height int) string {
	chart := &t.chartActivity
	borderColor := colorBorder
	if t.currentPane == 3 {
		borderColor = colorAccent
	}

	lines := make([]string, 0, len(chart.chartActivity.ChartLines)+1+len(chart.chartActivity.Activities))
	for _, chartLine := range chart.chartActivity.ChartLines {
		lines = append(lines, fg(colorAccent).Render(chartLine))
	}
	lines = append(lines, "")
	for _, activity := range chart.chartActivity.Activities {
		lines = append(lines,
			fg(colorMuted).Render(activity.Time)+" "+
				fg(colorWhite).Render(activity.Label)+" "+
				fg(activityKindColor(activity.Kind)).Render(activity.Value))
	}
	return pane(chart.titleLabel, "", width, height, borderColor, lines)
}

func (t *topModel) renderKeyBindings(width int) string {
	muted := fg(colorMuted)
	lines := []string{
		muted.Render("s search   1-3 focus   j/k move   →/enter markets   ←/esc events   ? help"),
		muted.Render(t.commandPopup.statusLabel),
	}
	// Height 3 leaves a single visible content row, as in the Rust layout.
	return pane("Keybindings: ", "", width, 3, colorBorder, lines)
}
