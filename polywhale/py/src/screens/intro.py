from textual.app import ComposeResult
from textual.containers import Center, Horizontal, Middle
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static


class IntroScreen(Screen):

    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                yield Static("Polytop - Polymarket Monitor", id="title")
                yield Static("Hello, world!", id="body")
                with Horizontal():
                    yield Button("Dashboard", id="dashboard-btn", variant="primary")
                    yield Button("Settings", id="settings-btn")
        yield Footer()

    def action_go_dashboard(self) -> None:
        self.app.push_screen("dashboard")

    def action_go_settings(self) -> None:
        self.app.push_screen("settings")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "dashboard-btn":
            self.action_go_dashboard()
        elif event.button.id == "settings-btn":
            self.action_go_settings()
