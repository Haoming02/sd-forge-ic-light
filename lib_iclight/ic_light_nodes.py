from typing import Callable, Optional, TypedDict

import torch

from modules.devices import device, dtype

try:
    from ldm_patched.modules.model_patcher import ModelPatcher
except ImportError:
    from backend.patcher.base import ModelPatcher


class UnetParams(TypedDict):
    input: torch.Tensor
    timestep: torch.Tensor
    c: dict
    cond_or_uncond: torch.Tensor


class ICLight:
    """IC-Light Implementation"""

    @staticmethod
    def apply(
        model: ModelPatcher,
        ic_model_state_dict: dict[str, torch.Tensor],
        c_concat: dict,
        mode: Optional[str] = None,
    ) -> ModelPatcher:
        work_model = model.clone()

        model_config = (
            work_model.model.model_config
            if hasattr(work_model.model, "model_config")
            else work_model.model.config
        )
        scale_factor: float = model_config.latent_format.scale_factor

        concat_conds: torch.Tensor = c_concat["samples"] * scale_factor
        concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

        def apply_c_concat(params: UnetParams) -> UnetParams:
            """Apply c_concat on Unet call"""
            sample = params["input"]
            params["c"]["c_concat"] = torch.cat(
                (
                    [concat_conds.to(sample.device)]
                    * (sample.shape[0] // concat_conds.shape[0])
                ),
                dim=0,
            )
            return params

        def unet_dummy_apply(unet_apply: Callable, params: UnetParams) -> Callable:
            """A dummy unet apply wrapper serving as the endpoint of wrapper chain"""
            return unet_apply(x=params["input"], t=params["timestep"], **params["c"])

        existing_wrapper = work_model.model_options.get(
            "model_function_wrapper", unet_dummy_apply
        )

        def wrapper_func(unet_apply: Callable, params: UnetParams) -> Callable:
            return existing_wrapper(unet_apply, params=apply_c_concat(params))

        work_model.set_model_unet_function_wrapper(wrapper_func)

        args = {
            "patches": {
                ("diffusion_model." + key): (value.to(dtype=dtype, device=device),)
                for key, value in ic_model_state_dict.items()
            }
        }

        if mode is not None:
            args["filename"] = f"ic-light-{mode}"

        work_model.add_patches(**args)
        return work_model
