# ============================================================= #
# Reference:                                                    #
# https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg #
# ============================================================= #

import os

import numpy as np
import rembg
from PIL import Image

from modules.paths import models_path
from modules.shared import opts

if "U2NET_HOME" not in os.environ:
    os.environ["U2NET_HOME"] = os.path.join(models_path, "u2net")


def get_models() -> tuple[str]:
    if getattr(opts, "ic_all_rembg", False):
        return (
            "u2net",
            "u2netp",
            "u2net_human_seg",
            "u2net_cloth_seg",
            "isnet-anime",
            "isnet-general-use",
            "silueta",
        )
    else:
        return (
            "u2net_human_seg",
            "isnet-anime",
        )


def run_rmbg(
    np_image: np.ndarray,
    model: str,
    foreground_threshold: int,
    background_threshold: int,
    erode_size: int,
) -> np.ndarray:
    image = Image.fromarray(np_image.astype(np.uint8)).convert("RGB")

    processed_image = rembg.remove(
        image,
        session=rembg.new_session(
            model_name=model,
            providers=["CPUExecutionProvider"],
        ),
        alpha_matting=True,
        alpha_matting_foreground_threshold=foreground_threshold,
        alpha_matting_background_threshold=background_threshold,
        alpha_matting_erode_size=erode_size,
        post_process_mask=True,
        only_mask=False,
        bgcolor=(127, 127, 127, 255),
    )

    return np.asarray(processed_image.convert("RGB"), dtype=np.uint8)
