# =========================================== #
# Reference:                                  #
# https://youtu.be/5EuYKEvugLU?feature=shared #
# =========================================== #

from modules.images import resize_image
from PIL import Image
import numpy as np
import cv2


def resize_input(img: np.ndarray, h: int, w: int, mode: int = 1) -> np.ndarray:
    img = Image.fromarray(img)
    resized_img: Image = resize_image(mode, img, w, h)  # Crop & Resize

    return np.asarray(resized_img.convert("RGB")).astype(np.uint8)


def restore_detail(
    ic_light_image: np.ndarray,
    original_image: np.ndarray,
    blur_radius: int = 5,
) -> Image:

    h, w, c = ic_light_image.shape
    original_image = resize_input(original_image, h, w)

    if c == 4:
        ic_light_image = cv2.cvtColor(ic_light_image, cv2.COLOR_RGBA2RGB)

    ic_light_image = ic_light_image.astype(np.float32) / 255.0
    original_image = original_image.astype(np.float32) / 255.0

    blurred_ic_light = cv2.GaussianBlur(ic_light_image, (blur_radius, blur_radius), 0)
    blurred_original = cv2.GaussianBlur(original_image, (blur_radius, blur_radius), 0)

    DoG = original_image - blurred_original + blurred_ic_light
    DoG = np.clip(DoG * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(DoG)
