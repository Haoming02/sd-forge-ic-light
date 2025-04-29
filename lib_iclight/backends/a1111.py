from ..logging import logger

try:
    from lib_modelpatcher.model_patcher import ModulePatch
except ImportError:
    logger.error("Please install [sd-webui-model-patcher] first!")
    raise

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

    from ..parameters import ICLightArgs

from functools import wraps

import safetensors.torch
import torch

from modules.devices import device, dtype

from ..model_loader import ICModels
from ..utils import numpy2pytorch


def vae_encode(sd_model, image: torch.Tensor) -> torch.Tensor:
    """
    image: [B, C, H, W] format tensor, ranging from -1.0 to 1.0
    Return: tensor in [B, C, H, W] format

    Note: Input image format differs from Forge/Comfy's VAE input format
    """
    return sd_model.get_first_stage_encoding(sd_model.encode_first_stage(image))


@torch.inference_mode()
def apply_ic_light(p: "StableDiffusionProcessing", args: "ICLightArgs"):
    sd = safetensors.torch.load_file(ICModels.get_path(args.model_type))

    concat_conds = vae_encode(
        p.sd_model,
        numpy2pytorch(args.get_concat_cond(p)).to(dtype=dtype, device=device),
    ).to(dtype=dtype)

    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    def apply_c_concat(unet, old_forward: Callable) -> Callable:
        @wraps(old_forward)
        def new_forward(x, timesteps=None, context=None, **kwargs):
            c_concat = torch.cat(
                ([concat_conds.to(x.device)] * (x.shape[0] // concat_conds.shape[0])),
                dim=0,
            )
            new_x = torch.cat([x, c_concat], dim=1)
            return old_forward(new_x, timesteps, context, **kwargs)

        return new_forward

    model_patcher = p.get_model_patcher()
    model_patcher.add_module_patch(
        "diffusion_model",
        ModulePatch(create_new_forward_func=apply_c_concat),
    )
    model_patcher.add_patches(
        patches={
            "diffusion_model." + key: (value.to(dtype=dtype, device=device),)
            for key, value in sd.items()
        }
    )
