from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.events import Key

from screens.dashboard import DashboardScreen
from screens.intro import IntroScreen
from screens.settings import SettingsScreen


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

    def on_key(self, event: Key) -> None:
        pass


def main() -> None:
    PolytopApp().run()
