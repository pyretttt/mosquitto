package tui

import (
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

type command int

const (
	commandHelp command = iota
	commandQuit
	commandIntro
	commandLog
)

var allCommands = []command{commandHelp, commandQuit, commandIntro, commandLog}

func (c command) name() string {
	switch c {
	case commandHelp:
		return "help"
	case commandQuit:
		return "quit"
	case commandIntro:
		return "intro"
	default:
		return "log"
	}
}

func (c command) description() string {
	switch c {
	case commandHelp:
		return "Show help"
	case commandQuit:
		return "Quit the application"
	case commandIntro:
		return "Show intro"
	default:
		return "Show logs"
	}
}

func commandFromName(name string) (command, bool) {
	for _, cmd := range allCommands {
		if cmd.name() == name {
			return cmd, true
		}
	}
	return 0, false
}

type paletteActionKind int

const (
	paletteActionNone paletteActionKind = iota
	paletteActionClose
	paletteActionRun
)

type paletteAction struct {
	kind    paletteActionKind
	command command
}

type paletteModel struct {
	text        string
	placeholder string
	selected    int
}

func newPaletteModel() paletteModel {
	return paletteModel{placeholder: "Type a command"}
}

func (p *paletteModel) availableCommands() []command {
	matches := make([]command, 0, len(allCommands))
	for _, cmd := range allCommands {
		if strings.HasPrefix(cmd.name(), p.text) {
			matches = append(matches, cmd)
		}
	}
	return matches
}

func (p *paletteModel) selectedCommand() (command, bool) {
	available := p.availableCommands()
	if p.selected >= 0 && p.selected < len(available) {
		return available[p.selected], true
	}
	return 0, false
}

func (p *paletteModel) commandToComplete() (command, bool) {
	if p.text == "" {
		return 0, false
	}
	if available := p.availableCommands(); len(available) > 0 {
		return available[0], true
	}
	return 0, false
}

func (p *paletteModel) changeText(text string) {
	p.text = text
	p.selected = 0
}

// handleKey ports command_pallete_key_input_middleware.
func (p *paletteModel) handleKey(msg tea.KeyMsg) (bool, paletteAction) {
	none := paletteAction{kind: paletteActionNone}
	switch msg.String() {
	case "ctrl+c":
		return false, none

	case "ctrl+w":
		words := strings.Split(p.text, " ")
		if len(words) > 0 {
			p.changeText(strings.Join(words[:len(words)-1], " "))
		}
		return true, none

	case "backspace":
		if runes := []rune(p.text); len(runes) > 0 {
			p.changeText(string(runes[:len(runes)-1]))
		} else {
			p.changeText("")
		}
		return true, none

	case "up":
		p.selected = max(p.selected-1, 0)
		return true, none

	case "down":
		p.selected = min(p.selected+1, max(len(p.availableCommands())-1, 0))
		return true, none

	case "esc":
		return true, paletteAction{kind: paletteActionClose}

	case "tab":
		if cmd, ok := p.commandToComplete(); ok {
			p.text = cmd.name()
		}
		return true, none

	case "enter":
		if cmd, ok := commandFromName(p.text); ok {
			return true, paletteAction{kind: paletteActionRun, command: cmd}
		}
		if p.text == "" {
			if cmd, ok := p.selectedCommand(); ok {
				p.text = cmd.name()
			}
		}
		return true, none

	default:
		if msg.Type == tea.KeyRunes && !msg.Alt {
			p.changeText(p.text + string(msg.Runes))
			return true, none
		}
		return false, none
	}
}

// view renders the bottom-anchored popup: a command table plus input line.
func (p *paletteModel) view(width, height int) string {
	popupH := clampInt(height*2/3, 8, 18)
	popupH = min(popupH, height)
	innerH := popupH - 2

	// Bottom 3 rows of the popup hold the input (separator + text line).
	commandsH := max(innerH-3, 0)

	available := p.availableCommands()
	selected := min(p.selected, max(len(available)-1, 0))

	var lines []string
	lines = append(lines, "") // vertical margin above the table
	if len(available) == 0 {
		lines = append(lines, "  "+fg(colorMuted).Render("No commands"))
	}
	for i, cmd := range available {
		nameStyle, descStyle := fg(colorWhite), fg(colorMuted)
		if i == selected {
			nameStyle, descStyle = fgBold(colorFocus), fgBold(colorFocus)
		}
		lines = append(lines, "  "+tableRow([]int{12, 40}, []tableCell{
			{text: cmd.name(), style: nameStyle},
			{text: " " + cmd.description(), style: descStyle},
		}))
	}
	if len(lines) > commandsH+1 {
		lines = lines[:commandsH+1]
	}
	for len(lines) < commandsH+1 {
		lines = append(lines, "")
	}

	lines = append(lines, fg(colorGray).Render(strings.Repeat("─", width-2)))
	lines = append(lines, " "+p.inputLine())

	return pane(" Commands ", "", width, popupH, colorWhite, lines)
}

func (p *paletteModel) inputLine() string {
	if p.text == "" {
		return fg(colorMuted).Render(p.placeholder)
	}
	line := fg(colorWhite).Render(p.text)
	if cmd, ok := p.commandToComplete(); ok {
		line += fg(colorMuted).Render(cmd.name()[len(p.text):])
	}
	return line
}
