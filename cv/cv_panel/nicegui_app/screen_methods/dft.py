from dataclasses import replace

import numpy as np
import PIL.Image as PILImage


from nicegui_app.state import Screen, Option, NumberFieldOption

AMPLITUDE_SCALE_ID = 0
PHASE_SCALE_ID = 1


def transform(input: np.array, amplitude_scale: int, phase_scale: int):
    f = np.fft.fft2(input)
    fshift = np.fft.fftshift(f)

    amplitude = np.abs(fshift)
    amplitude = amplitude_scale * np.log1p(amplitude)
    phase = phase_scale * np.angle(fshift)

    return amplitude, phase


def run(screen: Screen) -> Screen:
    amplitude_scale = screen.option_with_id(AMPLITUDE_SCALE_ID).info.value
    phase_scale: int = screen.option_with_id(PHASE_SCALE_ID).info.value
    input = np.array(screen.workspace_state.input[0].convert("L"))

    amplitude, phase = transform(input, amplitude_scale, phase_scale)

    return replace(
        screen,
        workspace_state=replace(screen.workspace_state, output=[PILImage.fromarray(amplitude).convert("L")]),
    )


screen = Screen(
    name="Discrete Fourier Transform",
    description="Visualize Fourier transform amplitude and phase component",
    options=[
        Option(
            name="Amplitude scale",
            info=NumberFieldOption(value=20, min_value=0, max_value=1000),
            description="Scale of output amplitude matrix",
            id=AMPLITUDE_SCALE_ID,
        ),
        Option(
            name="Phase scale",
            info=NumberFieldOption(value=255, min_value=0, max_value=255),
            description="Scale of output phase matrix",
            id=PHASE_SCALE_ID,
        ),
    ],
    run=run,
)
