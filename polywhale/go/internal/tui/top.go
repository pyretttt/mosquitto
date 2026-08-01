package tui

import (
	"context"
	"fmt"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"polywhale/internal/logging"
	"polywhale/internal/polymarket"
)

const (
	topPageTitle      = " POLYWHALE "
	searchPlaceholder = "search by slug/tags/id"

	// Max visible rows in the markets table (scrolls when there are more).
	maxMarketsTableHeight = 6
)

type statusPane struct {
	isOnline        bool
	wsLive          bool
	latency         uint64
	refreshInterval uint64
	mode            string

	latencyLabel string
	refreshLabel string
	modeLabel    string
}

func (s *statusPane) refreshLabels() {
	s.latencyLabel = fmt.Sprintf("   latency %dms", s.latency)
	s.refreshLabel = fmt.Sprintf("   refresh %dms", s.refreshInterval)
	s.modeLabel = fmt.Sprintf("   mode %s", s.mode)
}

type searchState struct {
	term        string
	placeholder string
	focused     bool
	page        int
	searchType  polymarket.SearchType
}

func emptySearch() *searchState {
	return &searchState{
		placeholder: searchPlaceholder,
		focused:     true,
		page:        1,
		searchType:  polymarket.SearchQuery,
	}
}

type eventsPane struct {
	title       string
	search      *searchState
	titleLabel  string
	footerLabel string
	data        polymarket.EventsData

	selected        int // selection index within the visible window
	marketsSelected int
	marketsFocused  bool

	windowSize int
	offset     int
}

func (p *eventsPane) refreshLabels() {
	searchLabel := "all"
	if p.search != nil {
		searchLabel = p.search.term
	}
	p.titleLabel = fmt.Sprintf(" [1] - %s: %s ", p.title, searchLabel)
	if p.marketsFocused {
		p.footerLabel = fmt.Sprintf(" %d events | j/k markets | ←/esc events | enter focus ", len(p.data.Events))
	} else {
		p.footerLabel = fmt.Sprintf(" %d events | j/k events | →/enter markets | b bookmarks ", len(p.data.Events))
	}
}

func (p *eventsPane) setSearch(search *searchState) {
	p.search = search
	p.refreshLabels()
}

func (p *eventsPane) selectedEventIdx() (int, bool) {
	abs := p.selected + p.offset
	if abs < len(p.data.Events) {
		return abs, true
	}
	return 0, false
}

func (p *eventsPane) selectedEvent() *polymarket.Event {
	if idx, ok := p.selectedEventIdx(); ok {
		return &p.data.Events[idx]
	}
	return nil
}

func (p *eventsPane) eventSlice() []polymarket.Event {
	if p.offset >= len(p.data.Events) {
		return nil
	}
	end := min(p.offset+p.windowSize, len(p.data.Events))
	return p.data.Events[p.offset:end]
}

// marketsTableHeight is the visible height of the nested markets table.
func (p *eventsPane) marketsTableHeight() int {
	if event := p.selectedEvent(); event != nil {
		return min(len(event.Markets), maxMarketsTableHeight)
	}
	return 0
}

func (p *eventsPane) clearMarketFocus() {
	p.marketsFocused = false
	p.marketsSelected = 0
	p.refreshLabels()
}

func (p *eventsPane) enterMarketFocus() bool {
	if p.marketsFocused {
		return true
	}
	event := p.selectedEvent()
	if event == nil || len(event.Markets) == 0 {
		return false
	}
	p.marketsFocused = true
	p.marketsSelected = 0
	p.refreshLabels()
	return true
}

type marketSummaryPane struct {
	title          string
	selectedMarket polymarket.SelectedMarket
}

type chartActivityPane struct {
	title         string
	titleLabel    string
	chartActivity polymarket.ChartActivity
}

type commandPopupPane struct {
	filter      string
	sort        string
	mode        string
	statusLabel string
}

func (c *commandPopupPane) refreshLabels() {
	c.statusLabel = fmt.Sprintf("Filter: %s sort:%s                               %s", c.filter, c.sort, c.mode)
}

type errorBanner struct {
	text  string
	token string
}

type topModel struct {
	leftTitle   string
	errorMsg    *errorBanner
	currentPane int

	status         statusPane
	events         eventsPane
	selectedMarket marketSummaryPane
	chartActivity  chartActivityPane
	commandPopup   commandPopupPane

	loadSession string // "" means no in-flight request
	isLoading   bool

	width  int
	height int
}

func eventsWindowSizeFor(height int) int {
	return clampInt(int(float64(height)*0.4), 10, 45)
}

// newTopModel ports TopPage::mock_data.
func newTopModel(width, height int) topModel {
	top := topModel{
		leftTitle:   topPageTitle,
		currentPane: 1,
		status: statusPane{
			isOnline:        true,
			wsLive:          true,
			latency:         42,
			refreshInterval: 500,
			mode:            "observe",
		},
		events: eventsPane{
			title:      "Top Events",
			windowSize: eventsWindowSizeFor(height),
		},
		selectedMarket: marketSummaryPane{
			title: "Will BTC hit 100k in 2026?",
			selectedMarket: polymarket.NewSelectedMarket(
				"btc-100k-2026",
				63.0, 38.0, 62.0, 64.0, 37.0, 39.0, 2.0,
				842_100.0, 184_300.0, 2_400_000.0,
				"2026-12-31",
			),
		},
		chartActivity: chartActivityPane{
			title:      "Chart + Activity",
			titleLabel: " [3] - Chart + Activity ",
			chartActivity: polymarket.ChartActivity{
				ChartLines: []string{
					" 70¢ ┤                     ╭╮",
					" 65¢ ┤              ╭──────╯╰─╮",
					" 60¢ ┤      ╭───────╯         ╰╮",
					" 55¢ ┤ ╭────╯                  ╰─",
					"     └────────────────────────────",
				},
				Activities: []polymarket.ActivityEntry{
					{Time: "17:42", Label: "price", Value: "+4¢", Kind: polymarket.ActivityPositive},
					{Time: "17:41", Label: "best bid", Value: "62¢", Kind: polymarket.ActivityAccent},
					{Time: "17:40", Label: "trade", Value: "219 @45¢", Kind: polymarket.ActivityWarning},
					{Time: "17:39", Label: "spread", Value: "2¢", Kind: polymarket.ActivityMuted},
				},
			},
		},
		commandPopup: commandPopupPane{
			filter: "politics volume>100k",
			sort:   "move",
			mode:   "NORMAL",
		},
		width:  width,
		height: height,
	}
	top.events.refreshLabels()
	top.commandPopup.refreshLabels()
	top.status.refreshLabels()
	return top
}

func (t *topModel) updateWindowSize(width, height int) {
	t.width, t.height = width, height
	t.events.windowSize = eventsWindowSizeFor(height)
}

func (t *topModel) tableHeight() int {
	const topTablePayloadHeightAddend = 3
	eventsH := min(len(t.events.eventSlice()), t.events.windowSize)
	return max(eventsH+t.events.marketsTableHeight(), 1) + topTablePayloadHeightAddend
}

func (t *topModel) setEventsLoadingSession(token string) {
	t.loadSession = token
	t.isLoading = token != ""
}

// resetForSearch ports TopPage::set_search.
func (t *topModel) resetForSearch(clearSearch bool) {
	t.events.data = polymarket.EventsData{}
	t.events.offset = 0
	t.events.selected = 0
	t.events.clearMarketFocus()
	if clearSearch {
		t.events.setSearch(nil)
	}
	t.setEventsLoadingSession("")
}

// requestEvents starts an async page load unless one is already in flight.
func (t *topModel) requestEvents(client *polymarket.Client) tea.Cmd {
	if t.loadSession != "" {
		return nil
	}
	session := genToken()
	t.setEventsLoadingSession(session)

	cursor := t.events.data.NextCursor
	var filter *polymarket.EventsFilter
	if s := t.events.search; s != nil {
		filter = &polymarket.EventsFilter{
			SearchType: s.searchType,
			Query:      s.term,
			Page:       s.page,
		}
	}
	return func() tea.Msg {
		data, err := client.LoadEvents(context.Background(), cursor, filter)
		if err != nil {
			logging.Errorf("app", "[TopPage] EventsRequestFailed: %v", err)
			return eventsLoadFailedMsg{session: session}
		}
		return eventsLoadedMsg{data: data, session: session}
	}
}

func (t *topModel) applyEventsLoaded(msg eventsLoadedMsg) {
	if t.loadSession == msg.session {
		startRank := len(t.events.data.Events) + 1
		wasEmpty := len(t.events.data.Events) == 0
		for i := range msg.data.Events {
			msg.data.Events[i].SetRank(startRank + i)
		}
		t.events.data.Events = append(t.events.data.Events, msg.data.Events...)
		t.events.data.NextCursor = msg.data.NextCursor
		t.events.refreshLabels()
		if wasEmpty && len(t.events.data.Events) > 0 {
			t.events.selected = 0
		}
	}
	t.setEventsLoadingSession("")
}

func (t *topModel) applyEventsLoadFailed() tea.Cmd {
	token := genToken()
	t.errorMsg = &errorBanner{
		text:  "Failed to load events data, press `R` to retry",
		token: token,
	}
	t.setEventsLoadingSession("")
	return tea.Tick(3*time.Second, func(time.Time) tea.Msg {
		return hideErrorMsg{token: token}
	})
}

func (t *topModel) hideError(token string) {
	if t.errorMsg != nil && t.errorMsg.token == token {
		t.errorMsg = nil
	}
}

// moveEventSelection moves the selection up/down, scrolling the window and
// requesting the next page when moving past the last loaded event.
func (t *topModel) moveEventSelection(down bool, client *polymarket.Client) tea.Cmd {
	total := len(t.events.data.Events)
	selectedInWindow := t.events.selected
	abs := selectedInWindow + t.events.offset

	var cmd tea.Cmd
	if down {
		if abs >= total-1 {
			if t.isLoading {
				return nil
			}
			logging.Infof("app", "TopPage: Loading more events")
			if t.events.search != nil {
				t.events.search.page++
			}
			return t.requestEvents(client)
		}
		if selectedInWindow < t.events.windowSize-1 {
			t.events.selected++
		} else {
			t.events.offset++
		}
	} else {
		if abs == 0 {
			return nil
		}
		if selectedInWindow != 0 {
			t.events.selected--
		} else {
			t.events.offset = max(t.events.offset-1, 0)
		}
	}
	t.events.clearMarketFocus()
	return cmd
}

// handleKey ports TopPage::key_input_middleware. Returns whether the key was
// consumed.
func (t *topModel) handleKey(msg tea.KeyMsg, client *polymarket.Client) (bool, tea.Cmd) {
	if t.events.search != nil && t.events.search.focused {
		return t.handleSearchKey(msg, client)
	}

	switch key := msg.String(); key {
	case "1", "2", "3":
		t.currentPane = int(key[0] - '0')
		return true, nil

	case "enter", "right":
		t.events.enterMarketFocus()
		return true, nil

	case "left", "esc":
		if t.events.marketsFocused {
			t.events.clearMarketFocus()
			return true, nil
		}
		return false, nil

	case "s":
		if t.events.search != nil {
			t.events.search.focused = true
		} else {
			t.events.setSearch(emptySearch())
		}
		return true, nil

	case "j", "k":
		if len(t.events.data.Events) == 0 {
			return false, nil
		}
		down := key == "j"
		if t.events.marketsFocused {
			marketCount := 0
			if event := t.events.selectedEvent(); event != nil {
				marketCount = len(event.Markets)
			}
			marketIdx := t.events.marketsSelected
			if down {
				if marketIdx >= marketCount-1 {
					// Edge: leave markets and move to next event.
					t.events.clearMarketFocus()
					return true, t.moveEventSelection(true, client)
				}
				t.events.marketsSelected++
			} else {
				if marketIdx == 0 {
					t.events.clearMarketFocus()
					return true, t.moveEventSelection(false, client)
				}
				t.events.marketsSelected--
			}
			return true, nil
		}
		return true, t.moveEventSelection(down, client)
	}
	return false, nil
}

func (t *topModel) handleSearchKey(msg tea.KeyMsg, client *polymarket.Client) (bool, tea.Cmd) {
	search := t.events.search
	switch msg.String() {
	case "tab":
		search.searchType = search.searchType.Next()
		return true, nil

	case "esc":
		t.resetForSearch(true)
		return true, t.requestEvents(client)

	case "enter":
		search.focused = false
		t.resetForSearch(false)
		return true, t.requestEvents(client)

	case "backspace":
		if runes := []rune(search.term); len(runes) > 0 {
			search.term = string(runes[:len(runes)-1])
		}
		return true, nil

	default:
		if msg.Type == tea.KeyRunes && !msg.Alt {
			search.term += string(msg.Runes)
		}
		// Swallow everything else while the search input is focused.
		return true, nil
	}
}
