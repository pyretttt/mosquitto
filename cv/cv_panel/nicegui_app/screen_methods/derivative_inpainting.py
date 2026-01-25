from dataclasses import replace

from nicegui_app.state import Screen, Option, NumberFieldOption

from PIL.Image import Image as PILImage
import numpy as np

X_ID = 0
Y_ID = 1
WIDTH_ID = 2
HEIGHT_ID = 3


def transform(input: PILImage, x: float, y: float, width: float, height: float) -> PILImage:
    pass


def run(screen: Screen) -> Screen:
    input = np.array(screen.workspace_state.input[0])
    x = screen.option_with_id(X_ID).info.value
    y = screen.option_with_id(Y_ID).info.value
    width = screen.option_with_id(WIDTH_ID).info.value
    height = screen.option_with_id(HEIGHT_ID).info.value

    input, output = transform(input, x, y, width, height)

    return replace(
        screen,
        workspace_state=replace(
            screen.workspace_state,
            input=[PILImage.fromarray(input)],
            output=[PILImage.fromarray(output)],
        ),
    )


screen = Screen(
    name="Derivative inpainting",
    description="Inpainting using derivative of the image",
    options=[
        Option(
            name="X",
            description="top left x coordinate of the area to inpaint",
            info=NumberFieldOption(value=0.0, min_value=0.0, max_value=1.0),
            id=X_ID,
        ),
        Option(
            name="Y",
            description="top left y coordinate of the area to inpaint",
            info=NumberFieldOption(value=0.0, min_value=0.0, max_value=1.0),
            id=Y_ID,
        ),
        Option(
            name="Width",
            description="width of the area to inpaint",
            info=NumberFieldOption(value=0.0, min_value=0.0, max_value=1.0),
            id=WIDTH_ID,
        ),
        Option(
            name="Height",
            description="height of the area to inpaint",
            info=NumberFieldOption(value=0.0, min_value=0.0, max_value=1.0),
            id=HEIGHT_ID,
        ),
    ],
    run=run,
)
