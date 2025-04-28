import numpy as np
import torch
from PIL import Image

from modules.images import LANCZOS


def numpy2pytorch(imgs: np.ndarray) -> torch.Tensor:
    """Automatic1111's VAE accepts -1.0 ~ 1.0 tensors"""
    h = torch.from_numpy(np.stack(imgs, axis=0, dtype=np.float32)) / 127 - 1.0
    h = h.movedim(-1, 1)
    return h


def forge_numpy2pytorch(img: np.ndarray) -> torch.Tensor:
    """Forge & ComfyUI's VAE accepts 0.0 ~ 1.0 tensors"""
    h = torch.from_numpy(img.astype(np.float32) / 255)
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image: np.ndarray, w: int, h: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(w / original_width, h / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), LANCZOS)
    left = (resized_width - w) / 2
    top = (resized_height - h) / 2
    right = (resized_width + w) / 2
    bottom = (resized_height + h) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.asarray(cropped_image, dtype=np.uint8)


def align_dim_latent(x: int) -> int:
    """
    Align the pixel dimension to latent dimension\n
    Stable Diffusion uses 1:8 ratio for latent:pixel\n
    i.e. 1 latent unit == 8 pixel unit
    """
    return round(x / 8) * 8


def make_masked_area_grey(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Make the masked area grey"""
    return (
        (image.astype(np.float32) * alpha + (1.0 - alpha) * 127)
        .round()
        .clip(0, 255)
        .astype(np.uint8)
    )
