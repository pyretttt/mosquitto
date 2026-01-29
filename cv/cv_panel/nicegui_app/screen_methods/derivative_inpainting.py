from dataclasses import replace

from nicegui_app.state import Screen, Option, NumberFieldOption

import PIL.Image as PILImage
import numpy as np
import cv2 as cv

X_ID = 0
Y_ID = 1
WIDTH_ID = 2
HEIGHT_ID = 3


def vectorize_image(image: np.array) -> np.array:
    """
    vectorizes grayscale input image info row vector of size H x W
    """
    return image.ravel()


def sobel_kernel_for_vectorized_input(image_shape: tuple[int, int], vec_image: np.array, kernel: np.array) -> np.array:
    """
    Returns matrix for Ax. which convolves vectorized image with sobel kernel
    """
    H, W = image_shape
    ksize = kernel.shape[0]
    sparse_kernel = np.zeros((len(vec_image) - ksize + 1, len(vec_image)))


def transform(input: np.array, x: float, y: float, width: float, height: float) -> tuple[np.array, np.array]:
    input = input[:, :, :3].copy()
    gray_scale_input = input.mean(axis=-1)

    H, W = input.shape[:2]
    x_min = int(x * W)
    y_min = int(y * H)
    x_max = x_min + int(width * W)
    y_max = y_min + int(height * H)
    input_with_bbox = cv.rectangle(input, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2, lineType=cv.LINE_8)

    dx = cv.Sobel(gray_scale_input, cv.CV_64F, 1, 0, ksize=3)
    dy = cv.Sobel(gray_scale_input, cv.CV_64F, 0, 1, ksize=3)
    output = dx + dy
    output = np.abs(output).astype(np.uint8)
    return input_with_bbox, output


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
