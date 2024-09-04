from backend.patcher.base import ModelPatcher
from backend.modules.k_model import KModel
from backend import memory_management

from typing import TypedDict, Callable
import torch


class UnetParams(TypedDict):
    input: torch.Tensor
    timestep: torch.Tensor
    c: dict
    cond_or_uncond: torch.Tensor


class ICLight:
    """IC-Light Implementation"""

    def apply(
        self,
        model: ModelPatcher,
        ic_model_state_dict: dict,
        c_concat: dict,
    ) -> ModelPatcher:

        device = memory_management.get_torch_device()
        dtype = model.model.computation_dtype
        work_model = model.clone()

        # Apply scale factor
        base_model: KModel = work_model.model
        scale_factor = base_model.config.latent_format.scale_factor

        # [B, 4, H, W]
        concat_conds: torch.Tensor = c_concat["samples"] * scale_factor
        # [1, 4 * B, H, W]
        concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

        def apply_c_concat(params: UnetParams) -> UnetParams:
            """Apply c_concat on unet call."""
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

        # Compose on existing `model_function_wrapper`
        existing_wrapper = work_model.model_options.get(
            "model_function_wrapper", unet_dummy_apply
        )

        def wrapper_func(unet_apply: Callable, params: UnetParams) -> Callable:
            return existing_wrapper(unet_apply, params=apply_c_concat(params))

        work_model.set_model_unet_function_wrapper(wrapper_func)

        work_model.add_patches(
            filename="ic-light",
            patches={
                ("diffusion_model." + key): (value.to(dtype=dtype, device=device),)
                for key, value in ic_model_state_dict.items()
            },
        )

        return work_model
