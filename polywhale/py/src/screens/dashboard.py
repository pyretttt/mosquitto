from textual.app import ComposeResult
from textual.containers import Center, Middle
from textual.screen import Screen
from textual.widgets import Footer, Header, Static


class DashboardScreen(Screen):
    BINDINGS = [
        ("escape", "back", "Back"),
        ("b", "back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            with Middle():
                yield Static("Dashboard", id="title")
                yield Static("Market monitoring will live here.", id="body")
        yield Footer()

    def action_back(self) -> None:
        self.app.pop_screen()
