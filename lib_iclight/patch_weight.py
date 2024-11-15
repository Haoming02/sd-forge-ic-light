"""
Credit: huchenlei
https://github.com/huchenlei/ComfyUI-layerdiffuse/blob/v0.1.0/layered_diffusion.py#L35

Modified by. Haoming02 to work with reForge
"""

from ldm_patched.modules.model_management import cast_to_device
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules import lora
from functools import wraps
from typing import Callable
import torch


def adjust_channel(func: Callable):
    """Patches weight application to accept multi-channel inputs"""

    @torch.inference_mode()
    @wraps(func)
    def calculate_weight(*args) -> torch.Tensor:
        weight = func(*args)

        if isinstance(args[0], list):
            patches, weight, key = args[0:3]
        else:
            assert isinstance(args[1], list)
            patches, weight, key = args[1:4]

        for p in patches:
            alpha = p[0]
            v = p[1]

            # The recursion call should be handled in the main func call.
            if isinstance(v, list):
                continue

            if len(v) == 1:
                patch_type = "diff"
            elif len(v) == 2:
                patch_type = v[0]
                v = v[1]

            if patch_type == "diff":
                w1 = v[0]
                if all(
                    (
                        alpha != 0.0,
                        w1.shape != weight.shape,
                        w1.ndim == weight.ndim == 4,
                    )
                ):
                    new_shape = [max(n, m) for n, m in zip(weight.shape, w1.shape)]
                    print(
                        f"Merged with {key} channel changed from {weight.shape} to {new_shape}"
                    )
                    new_diff = alpha * cast_to_device(w1, weight.device, weight.dtype)
                    new_weight = torch.zeros(size=new_shape).to(weight)
                    new_weight[
                        : weight.shape[0],
                        : weight.shape[1],
                        : weight.shape[2],
                        : weight.shape[3],
                    ] = weight
                    new_weight[
                        : new_diff.shape[0],
                        : new_diff.shape[1],
                        : new_diff.shape[2],
                        : new_diff.shape[3],
                    ] += new_diff
                    new_weight = new_weight.contiguous().clone()
                    weight = new_weight

        return weight

    return calculate_weight


def patch():
    if hasattr(lora, "calculate_weight"):
        lora.calculate_weight = adjust_channel(lora.calculate_weight)
        print("\nlora.calculate_weight Patched!\n")
    else:
        ModelPatcher.calculate_weight = adjust_channel(ModelPatcher.calculate_weight)
        print("\nModelPatcher.calculate_weight Patched!\n")
