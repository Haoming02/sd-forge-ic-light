from modules.processing import StableDiffusionProcessing

from backend.memory_management import get_torch_device
from backend.patcher.base import ModelPatcher
from backend.utils import load_torch_file
from backend.patcher.vae import VAE

from .utils import forge_numpy2pytorch
from .ic_light_nodes import ICLight
from .args import ICLightArgs

import numpy as np
import torch


def apply_ic_light(
    p: StableDiffusionProcessing,
    args: ICLightArgs,
):
    device = get_torch_device()

    # Load model
    ic_model_state_dict = load_torch_file(args.model_type.path, device=device)

    # Get input
    input_fg_rgb: np.ndarray = args.input_fg_rgb

    # Apply IC Light
    work_model: ModelPatcher = p.sd_model.forge_objects.unet.clone()
    vae: VAE = p.sd_model.forge_objects.vae.clone()
    node = ICLight()

    # [B, C, H, W]
    pixel_concat = forge_numpy2pytorch(args.get_concat_cond(input_fg_rgb, p)).to(
        device=vae.device, dtype=torch.float16
    )

    # [B, H, W, C]
    # Forge/ComfyUI's VAE accepts [B, H, W, C] format
    pixel_concat = pixel_concat.movedim(1, 3)

    patched_unet: ModelPatcher = node.apply(
        model=work_model,
        ic_model_state_dict=ic_model_state_dict,
        c_concat={"samples": vae.encode(pixel_concat)},
        mode=args.model_type.name,
    )

    p.sd_model.forge_objects.unet = patched_unet

    # Add input image to extra result images
    if not getattr(p, "is_hr_pass", False):
        p.extra_result_images.append(input_fg_rgb)
