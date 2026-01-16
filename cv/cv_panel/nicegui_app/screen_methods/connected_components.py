from dataclasses import replace

import cv2
import numpy as np
import PIL.Image as PILImage
import matplotlib.cm as cm


from nicegui_app.state import Screen, NumberFieldOption, Option

NUM_OBJECTS_ID = 0


def get_connected_components(image: np.array, num_objects: int) -> np.array:
    cmap = cm.get_cmap("viridis", num_objects)
    colormap_array = cmap(np.linspace(0, 1, num_objects))[:, :3]

    offsets = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ]

    if len(image.shape) == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)

    H, W = image.shape[:2]
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    th = np.zeros((H + 2, W + 2))
    th[1 : H + 1, 1 : W + 1] = thresh[:, :, 0]

    visited = set()
    current_value = 0
    result = np.zeros((H + 2, W + 2, 3)).astype(np.float32)

    for row in range(1, H + 1):
        for col in range(1, W + 1):
            to_visit = [(row, col)]
            should_increment = False
            while len(to_visit):
                current_idx = to_visit.pop()
                if current_idx in visited:
                    continue
                visited.add(current_idx)
                if th[current_idx] == 0.0:
                    continue
                should_increment = True
                result[current_idx] = colormap_array[current_value % num_objects]
                for offset in offsets:
                    to_visit.append((current_idx[0] + offset[0], current_idx[1] + offset[1]))

            current_value += (0, 1)[should_increment]
    return (result * 255).astype(np.uint8)


def run(screen: Screen) -> Screen:
    input = screen.workspace_state.input[0]
    output = PILImage.fromarray(
        get_connected_components(image=np.array(input), num_objects=screen.option_with_id(NUM_OBJECTS_ID).info.value),
        mode="RGB",
    )
    return replace(screen, workspace_state=replace(screen.workspace_state, output=output))


screen = Screen(
    name="Connected components labeling",
    description="Segmentation based on connected components",
    options=[
        Option(
            name="Number of objects",
            description="Expected number of different objects",
            info=NumberFieldOption(value=20, min_value=0, max_value=1e7),
            id=NUM_OBJECTS_ID,
        )
    ],
    run=run,
)
