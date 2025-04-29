from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

    from ..parameters import ICLightArgs

try:
    from ldm_patched.modules.model_patcher import ModelPatcher
    from ldm_patched.modules.sd import VAE
    from ldm_patched.modules.utils import load_torch_file

    classic = True

except ImportError:
    from backend.patcher.base import ModelPatcher
    from backend.patcher.vae import VAE
    from backend.utils import load_torch_file

    classic = False

import torch

from modules.devices import device, dtype

from ..ic_light_nodes import ICLight
from ..model_loader import ICModels
from ..utils import forge_numpy2pytorch


@torch.inference_mode()
def apply_ic_light(p: "StableDiffusionProcessing", args: "ICLightArgs"):
    sd = load_torch_file(
        ICModels.get_path(args.model_type),
        safe_load=True,
        device=device,
    )

    work_model: ModelPatcher = p.sd_model.forge_objects.unet.clone()
    vae: VAE = p.sd_model.forge_objects.vae

    pixel_concat = (
        forge_numpy2pytorch(args.get_concat_cond(p))
        .to(device=vae.device, dtype=dtype)
        .movedim(1, 3)
    )

    patched_unet: ModelPatcher = ICLight.apply(
        model=work_model,
        ic_model_state_dict=sd,
        c_concat={"samples": vae.encode(pixel_concat)},
        mode=None if classic else args.model_type,
    )

    p.sd_model.forge_objects.unet = patched_unet
