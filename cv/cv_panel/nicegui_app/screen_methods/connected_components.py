from dataclasses import replace

from PIL.Image import Image

from nicegui_app.state import Screen

def process_input_image(image: Image) -> Image:
    pass


def run(screen: Screen) -> Screen:
    input = screen.workspace_state.input[0]
    return replace(
        screen,
        workspace_state=replace(
            screen.workspace_state,
            output=process_input_image(input)
        )
    )


screen = Screen(
    name="Connected components",
    description="Segmentation based on connected components",
    options=[

    ],
    run=run,
)