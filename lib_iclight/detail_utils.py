# =========================================== #
# Reference:                                  #
# https://youtu.be/5EuYKEvugLU?feature=shared #
# =========================================== #

from modules.images import resize_image

from PIL import Image
import numpy as np
import cv2


def resize_input(img: np.array, h: int, w: int, mode: int) -> np.array:
    img = Image.fromarray(img).convert("RGB")
    resized_img: Image = resize_image(mode, img, w, h)

    return np.asarray(resized_img).astype(np.uint8)


def restore_detail(
    ic_light_image: np.array,
    original_image: np.array,
    blur_radius: int = 5,
    resize_mode: int = 1,  # Crop & Resize
) -> Image:

    h, w, c = ic_light_image.shape
    original_image = resize_input(original_image, h, w, resize_mode)

    if len(ic_light_image.shape) == 2:
        ic_light_image = cv2.cvtColor(ic_light_image, cv2.COLOR_GRAY2RGB)
    elif ic_light_image.shape[2] == 4:
        ic_light_image = cv2.cvtColor(ic_light_image, cv2.COLOR_RGBA2RGB)

    assert ic_light_image.shape[2] == 3
    assert original_image.shape[2] == 3

    ic_light_image = ic_light_image.astype(np.float32) / 255.0
    original_image = original_image.astype(np.float32) / 255.0

    blurred_ic_light = cv2.GaussianBlur(ic_light_image, (blur_radius, blur_radius), 0)
    blurred_original = cv2.GaussianBlur(original_image, (blur_radius, blur_radius), 0)

    DoG = original_image - blurred_original + blurred_ic_light
    DoG = np.clip(DoG * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(DoG)
