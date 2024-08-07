from PIL import Image
import numpy as np
import torch


@torch.inference_mode()
def pytorch2numpy(imgs: torch.Tensor, quant: bool = True) -> list:
    results = []
    for x in imgs:
        y = x.movedim(0, -1).detach().float().cpu()

        if quant:
            y = y * 127.5 + 127.5
            y = y.numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs: np.ndarray) -> torch.Tensor:
    """Automatic1111's VAE accepts -1 ~ 1 tensors"""
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    h = h.movedim(-1, 1)
    return h


@torch.inference_mode()
def forge_numpy2pytorch(img: np.ndarray) -> torch.Tensor:
    """Forge & ComfyUI's VAE accepts 0 ~ 1 tensors"""
    h = torch.from_numpy(img.astype(np.float32) / 255.0)
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(
    image: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(
    image: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


def align_dim_latent(x: int) -> int:
    """
    Align the pixel dimension to latent dimension\n
    Stable Diffusion uses 1:8 ratio for latent:pixel\n
    i.e. 1 latent unit == 8 pixel unit
    """
    return (x // 8) * 8


def make_masked_area_grey(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Make the masked area grey"""
    return (
        (image.astype(np.float32) * alpha + (1 - alpha) * 127)
        .clip(0, 255)
        .astype(np.uint8)
    )
