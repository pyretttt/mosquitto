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
    if vec_image.ndim != 1 or vec_image.size != H * W:
        raise ValueError("vec_image must be a 1D array of length H*W")
    if kH > H or kW > W:
        raise ValueError("kernel must fit within image dimensions")

    outH = H - kH + 1
    outW = W - kW + 1
    dtype = np.result_type(kernel.dtype, np.float64)
    A = np.zeros((outH * outW, H * W), dtype=dtype)

    for out_idx in range(outH * outW):
        out_r = out_idx // outW
        out_c = out_idx % outW
        base = out_r * W + out_c  # top-left input index for this patch

        for kr in range(kH):
            row_start = base + kr * W
            A[out_idx, row_start : row_start + kW] = kernel[kr, :]
    return A


def pseudo_inverse(m: np.array):
    print("Before svd")

    U, S, V = np.linalg.svd(m)
    S = 1.0 / S
    print("Computed svd")
    return V @ S @ U.T


def transform(input: np.array, x: float, y: float, width: float, height: float) -> tuple[np.array, np.array]:
    input = input[::8, ::8, :3].copy()
    gray_scale_input = input.mean(axis=-1)

    H, W = gray_scale_input.shape[:2]
    kH, kW = SOBEL_Y.shape
    outH = H - kH + 1
    outW = W - kW + 1
    if outH <= 0 or outW <= 0:
        return input, gray_scale_input.astype(np.uint8)
    x_min = int(x * outW)
    y_min = int(y * outH)
    x_max = x_min + int(width * outW)
    y_max = y_min + int(height * outH)
    x_min = max(0, min(outW, x_min))
    y_min = max(0, min(outH, y_min))
    x_max = max(x_min, min(outW, x_max))
    y_max = max(y_min, min(outH, y_max))

    vectorized_grayscale = vectorize_image(gray_scale_input)
    conv_matrix = kernel_for_vectorized_img(gray_scale_input.shape, vec_image=vectorized_grayscale, kernel=SOBEL_Y)
    output = np.matmul(
        conv_matrix,
        vectorized_grayscale.astype(np.float32),
    )
    output = output.reshape((outH, outW))
    output[y_min:y_max, x_min:x_max] = 0

    kernel_inv = np.linalg.pinv(conv_matrix)
    grayscale_inpainted = np.matmul(kernel_inv, output.ravel())

    output = np.abs(grayscale_inpainted).reshape(H, W)
    output = np.clip(output, 0, 255).astype(np.uint8)
    return cv.rectangle(input, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2, cv.LINE_8), output


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
