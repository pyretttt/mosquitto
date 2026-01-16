from dataclasses import replace

import cv2
import numpy as np
import PIL.Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from nicegui_app.state import Screen

cmap = cm.get_cmap('viridis', 1000000)
colormap_array = cmap(np.linspace(0, 1, 1000000))[:, :3]

def get_connected_components(image: np.array) -> np.array:
    offsets = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1)
    ]

    H, W = image.shape[:2]
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    th = np.zeros((H + 2, W + 2))
    th[1: H + 1, 1: W + 1] = thresh[:]

    visited = set()
    current_value = 0
    result = np.zeros((H + 2, W + 2, 3)).astype(np.int32)

    for row in range(1, H + 1):
        for col in range(1, W + 1):
            to_visit = [(row, col)]
            should_increment = False
            while len(to_visit):
                current_idx = to_visit.pop()
                if current_idx in visited:
                    continue
                visited.add(current_idx)
                if th[current_idx[0], current_idx[1]] == 0:
                    continue
                should_increment = True
                result[current_idx] = colormap_array[current_value]
                for offset in offsets:
                    n_row, n_col = row + offset[0], col + offset[1]
                    to_visit.append((n_row, n_col))

            current_value += [1, 0][should_increment]

    return result[1: H + 1, 1: W+1]

def run(screen: Screen) -> Screen:
    input = screen.workspace_state.input[0]
    output = PILImage.fromarray(get_connected_components(np.array(input)), mode="L")
    return replace(
        screen,
        workspace_state=replace(
            screen.workspace_state,
            output=output
            )
    )


screen = Screen(
    name="Connected components labeling",
    description="Segmentation based on connected components",
    options=[],
    run=run,
)
