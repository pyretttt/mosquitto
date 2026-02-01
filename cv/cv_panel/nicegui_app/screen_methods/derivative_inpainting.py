from dataclasses import replace

from nicegui_app.state import Screen, Option, NumberFieldOption

import PIL.Image as PILImage
import numpy as np
import cv2 as cv

X_ID = 0
Y_ID = 1
WIDTH_ID = 2
HEIGHT_ID = 3

SOBEL_Y = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]
)


def vectorize_image(image: np.array) -> np.array:
    """
    vectorizes grayscale input image info row vector of size H x W
    """
    return image.ravel()


def kernel_for_vectorized_img(
    image_shape: tuple[int, int],
    vec_image: np.array,
    kernel: np.array,
) -> np.array:
    """
    Returns matrix for Ax. which convolves vectorized image with sobel kernel
    """
    H, W = image_shape
    kH, kW = kernel.shape
    assert vec_image.size == H * W, "vec_image must have length H*W"

    outH = H - kH + 1
    outW = W - kW + 1
    A = np.zeros((outH * outW, H * W), dtype=kernel.dtype)

    for out_idx in range(outH * outW):
        out_r = out_idx // outW
        out_c = out_idx % outW
        base = out_r * W + out_c  # top-left input index for this patch

        for kr in range(kH):
            row_start = base + kr * W
            A[out_idx, row_start : row_start + kW] = kernel[kr, :]
    return A


def pseudo_inverse(m: np.array):
    U, S, V = np.linalg.svd(m, full_matrices=True)
    S = 1.0 / S
    return V, S, U.T


def transform(input: np.array, x: float, y: float, width: float, height: float) -> tuple[np.array, np.array]:
    input = input[:, :, :3].copy()
    gray_scale_input = input.mean(axis=-1)
    gray_scale_input = gray_scale_input[::4, ::4]

    H, W = input.shape[:2]
    x_min = int(x * W)
    y_min = int(y * H)
    x_max = x_min + int(width * W)
    y_max = y_min + int(height * H)
    input_with_bbox = cv.rectangle(input, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2, lineType=cv.LINE_8)

    vectorized_grayscale = vectorize_image(gray_scale_input)

    kernel = kernel_for_vectorized_img(
        gray_scale_input.shape, vec_image=vectorize_image(gray_scale_input), kernel=SOBEL_Y
    )

    output = np.matmul(
        kernel_for_vectorized_img(gray_scale_input.shape, vec_image=vectorize_image(gray_scale_input), kernel=SOBEL_Y),
        vectorized_grayscale,
    )

    output = np.abs(output).reshape(gray_scale_input.shape[0] - 2, gray_scale_input.shape[1] - 2).astype(np.uint8)
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

if __name__ == "__main__":
    kernel_for_vectorized_img(image_shape=(420, 320), vec_image=np.random.randn(420 * 320), kernel=SOBEL_Y)
