from textual.app import ComposeResult
from textual.containers import Center, Middle
from textual.screen import Screen
from textual.widgets import Footer, Header, Static


class SettingsScreen(Screen):
    BINDINGS = [
        ("escape", "back", "Back"),
        ("b", "back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            with Middle():
                yield Static("Settings", id="title")
                yield Static("Configuration options will live here.", id="body")
        yield Footer()

    def action_back(self) -> None:
        self.app.pop_screen()
