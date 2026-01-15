from dataclasses import replace

import cv2
import numpy as np
import PIL.Image as PILImage

from nicegui_app.state import Screen


def process_input_image(image: PILImage.Image) -> PILImage.Image:
    image = np.array(image)
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    print(np.unique(labels * 10))
    colors = np.stack((np.array([0, 0, 0]), np.random.randint(0, 256, (num_labels, 3))), axis=0)
    # return PILImage.fromarray((np.random.randn(200, 200, 3) * 255).astype(np.uint8), mode="RGB")
    return PILImage.fromarray((labels * 10).astype(np.uint8), mode="L")


def run(screen: Screen) -> Screen:
    input = screen.workspace_state.input[0]
    return replace(screen, workspace_state=replace(screen.workspace_state, output=process_input_image(input)))


screen = Screen(
    name="Connected components",
    description="Segmentation based on connected components",
    options=[],
    run=run,
)
