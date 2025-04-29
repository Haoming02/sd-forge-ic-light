# =========================================== #
# Reference:                                  #
# https://youtu.be/5EuYKEvugLU?feature=shared #
# =========================================== #

import cv2
import numpy as np
from PIL import Image

from modules.images import resize_image


def resize_input(img: np.ndarray, h: int, w: int, mode: int = 1) -> np.ndarray:
    img = Image.fromarray(img)
    resized_img: Image.Image = resize_image(mode, img, w, h)  # Crop & Resize

    return np.asarray(resized_img.convert("RGB"), dtype=np.uint8)


def restore_detail(
    ic_light_image: np.ndarray,
    original_image: np.ndarray,
    blur_radius: int,
) -> Image.Image:
    h, w, c = ic_light_image.shape
    if c == 4:
        ic_light_image = cv2.cvtColor(ic_light_image, cv2.COLOR_RGBA2RGB)

    original_image = resize_input(original_image, h, w)

    ic_light_image = ic_light_image.astype(np.float32) / 255.0
    original_image = original_image.astype(np.float32) / 255.0

    blurred_ic_light = cv2.GaussianBlur(ic_light_image, (blur_radius, blur_radius), 0)
    blurred_original = cv2.GaussianBlur(original_image, (blur_radius, blur_radius), 0)

    DoG = original_image + (blurred_ic_light - blurred_original)
    DoG = np.clip(DoG * 255.0, 0, 255).round().astype(np.uint8)

    return Image.fromarray(DoG)
