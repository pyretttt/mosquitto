from textual.app import App, ComposeResult
from textual.binding import Binding

from polytop.screens.dashboard import DashboardScreen
from polytop.screens.intro import IntroScreen
from polytop.screens.settings import SettingsScreen


class PolytopApp(App):
    """Textual TUI scaffold for monitoring Polymarket."""

    TITLE = "Polytop"
    SUBTITLE = "Polymarket Monitor"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    SCREENS = {
        "intro": IntroScreen,
        "dashboard": DashboardScreen,
        "settings": SettingsScreen,
    }

    def on_mount(self) -> None:
        self.push_screen("intro")


def main() -> None:
    PolytopApp().run()
