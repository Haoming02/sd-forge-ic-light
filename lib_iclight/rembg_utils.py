# ============================================================= #
# Reference:                                                    #
# https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg #
# ============================================================= #

from modules.paths import models_path
from PIL import Image
import numpy as np
import rembg
import os

BASIC_MODELS = (
    "u2net_human_seg",
    "isnet-anime",
)

ALL_MODELS = (
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "isnet-anime",
    "isnet-general-use",
    "silueta",
)

GREY = (127, 127, 127, 255)


def run_rmbg(
    np_image: np.ndarray,
    model: str,
    foreground_threshold: int,
    background_threshold: int,
    erode_size: int,
    bg: tuple = GREY,
) -> np.ndarray:

    if "U2NET_HOME" not in os.environ:
        os.environ["U2NET_HOME"] = os.path.join(models_path, "u2net")

    image = Image.fromarray(np_image.astype(np.uint8)).convert("RGB")

    processed_image = rembg.remove(
        image,
        session=rembg.new_session(
            model_name=model,
            providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
        ),
        alpha_matting=True,
        alpha_matting_foreground_threshold=foreground_threshold,
        alpha_matting_background_threshold=background_threshold,
        alpha_matting_erode_size=erode_size,
        post_process_mask=True,
        only_mask=False,
        bgcolor=bg,
    )

    return np.array(processed_image.convert("RGB")).astype(np.uint8)
