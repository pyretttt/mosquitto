package tui

import (
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/google/uuid"

	"polywhale/internal/config"
	"polywhale/internal/logging"
	"polywhale/internal/polymarket"
)

type page int

const (
	pageLoading page = iota
	pageTop
	pageIntro
	pageHelp
)

type overlayKind int

const (
	overlayNone overlayKind = iota
	overlayLog
	overlayWindowSize
)

const (
	requiredWindowWidth  = 120
	requiredWindowHeight = 40
)

// Model is the root Bubble Tea model; it plays the role of the Rust
// AppState + Env + app_reducer combination.
type Model struct {
	cfg    *config.Config
	client *polymarket.Client
	memLog *logging.Memory

	width  int
	height int

	page    page
	loading loadingModel
	top     topModel

	overlay overlayKind
	logPage logModel
	winSize windowSizeModel

	palette *paletteModel

	incrementToken string
}

func NewModel(cfg *config.Config, client *polymarket.Client, memLog *logging.Memory) Model {
	return Model{
		cfg:     cfg,
		client:  client,
		memLog:  memLog,
		page:    pageLoading,
		loading: newLoadingModel(),
	}
}

func (m Model) Init() tea.Cmd {
	return tickCmd(m.cfg.TickInterval())
}

func genToken() string { return uuid.NewString() }

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tickMsg:
		if m.page == pageLoading {
			m.loading.tick(m.cfg.TickRate)
		}
		return m, tickCmd(m.cfg.TickInterval())

	case tea.WindowSizeMsg:
		return m.handleResize(msg)

	case tea.KeyMsg:
		return m.handleKey(msg)

	case debounceElapsedMsg:
		if m.incrementToken != msg.token {
			return m, nil
		}
		// LoadingPageAction::Finished
		if m.page == pageLoading {
			m.loading.finished = true
			m.loading.progress = 1.0
			m.top = newTopModel(m.width, m.height)
			m.page = pageTop
			return m, m.top.requestEvents(m.client)
		}
		return m, nil

	case eventsLoadedMsg:
		if m.page == pageTop {
			m.top.applyEventsLoaded(msg)
		}
		return m, nil

	case eventsLoadFailedMsg:
		if m.page == pageTop {
			return m, m.top.applyEventsLoadFailed()
		}
		return m, nil

	case hideErrorMsg:
		if m.page == pageTop {
			m.top.hideError(msg.token)
		}
		return m, nil
	}
	return m, nil
}

func (m Model) handleResize(msg tea.WindowSizeMsg) (tea.Model, tea.Cmd) {
	logging.Infof("app", "App: Resize: %d, %d", msg.Width, msg.Height)
	m.width, m.height = msg.Width, msg.Height

	invalidSize := msg.Width < requiredWindowWidth || msg.Height < requiredWindowHeight
	switch {
	case m.overlay == overlayWindowSize:
		if invalidSize {
			m.winSize.currentWidth, m.winSize.currentHeight = msg.Width, msg.Height
		} else {
			m.overlay = overlayNone
		}
	case invalidSize:
		m.winSize = newWindowSizeModel(requiredWindowWidth, requiredWindowHeight)
		m.winSize.currentWidth, m.winSize.currentHeight = msg.Width, msg.Height
		m.overlay = overlayWindowSize
	}

	if m.page == pageTop {
		m.top.updateWindowSize(msg.Width, msg.Height)
	}
	return m, nil
}

func (m Model) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	// Middleware chain, in the same order as the Rust app_reducer.
	if m.palette != nil {
		if handled, model, cmd := m.paletteHandleKey(msg); handled {
			return model, cmd
		}
	}

	if m.overlay == overlayLog {
		if handled, closed := m.logPage.handleKey(msg); handled {
			if closed {
				m.overlay = overlayNone
			}
			return m, nil
		}
	}

	if m.page == pageTop {
		if handled, cmd := m.top.handleKey(msg, m.client); handled {
			return m, cmd
		}
	}

	switch msg.String() {
	case "esc", "q", "ctrl+c":
		return m, tea.Quit
	case "enter":
		// Debounce example: finish loading one second after Enter.
		token := genToken()
		m.incrementToken = token
		return m, tea.Tick(time.Second, func(time.Time) tea.Msg {
			return debounceElapsedMsg{token: token}
		})
	case "/":
		palette := newPaletteModel()
		m.palette = &palette
	}
	return m, nil
}

// paletteHandleKey routes keys into the command palette and applies the
// resulting action to the root model.
func (m Model) paletteHandleKey(msg tea.KeyMsg) (bool, tea.Model, tea.Cmd) {
	handled, action := m.palette.handleKey(msg)
	if !handled {
		return false, m, nil
	}
	switch action.kind {
	case paletteActionClose:
		m.palette = nil
	case paletteActionRun:
		m.palette = nil
		switch action.command {
		case commandHelp:
			m.page = pageHelp
		case commandQuit:
			return true, m, tea.Quit
		case commandIntro:
			m.page = pageIntro
		case commandLog:
			m.logPage = newLogModel(m.memLog)
			m.overlay = overlayLog
		}
	}
	return true, m, nil
}

func (m Model) View() string {
	if m.width == 0 || m.height == 0 {
		return ""
	}

	var view string
	switch m.page {
	case pageLoading:
		view = m.loading.view(m.width, m.height)
	case pageTop:
		view = m.top.view()
	case pageIntro:
		view = introView(m.width, m.height)
	case pageHelp:
		view = "" // The Rust Help page renders nothing.
	}

	switch m.overlay {
	case overlayLog:
		view = m.logPage.view(m.width, m.height)
	case overlayWindowSize:
		view = m.winSize.view(m.width, m.height)
	}

	if m.palette != nil {
		popup := m.palette.view(m.width, m.height)
		view = overlayBottom(view, popup, m.width, m.height)
	}
	return view
}

// introView ports the placeholder intro page verbatim.
func introView(width, height int) string {
	_ = height
	lines := []string{
		"",
		" " + fgBold(colorMagenta).Render("intro.title.as_str()"),
		" intro.text.as_str()",
		" counter: 0",
		" " + fg(colorMuted).Render("Press q to quit"),
	}
	out := ""
	for _, line := range lines {
		out += padLine(line, width) + "\n"
	}
	return out[:len(out)-1]
}
