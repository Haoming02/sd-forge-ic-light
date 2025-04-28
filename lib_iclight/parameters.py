import numpy as np
from PIL import Image

from modules.api.api import decode_base64_to_image
from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
    StableDiffusionProcessingTxt2Img,
)

from .logging import logger
from .model_loader import ICModels
from .rembg_utils import run_rmbg
from .utils import (
    align_dim_latent,
    make_masked_area_grey,
    resize_and_center_crop,
)


class DetailTransfer:
    def __init__(self, transfer: bool, radius: int, original: np.ndarray):
        self.enable = transfer
        self.radius = radius
        self.original = original


class ICLightArgs:
    def __init__(
        self,
        p: StableDiffusionProcessing,
        model_type: str,
        input_fg: np.ndarray,
        uploaded_bg: np.ndarray,
        remove_bg: bool,
        rembg_model: str,
        foreground_threshold: int,
        background_threshold: int,
        erode_size: int,
        detail_transfer: bool,
        detail_transfer_raw: bool,
        detail_transfer_blur_radius: int,
        reinforce_fg: bool,
    ):
        self.model_type: str = model_type

        if isinstance(p, StableDiffusionProcessingImg2Img):
            self.input_fg: np.ndarray = np.asarray(p.init_images[0], dtype=np.uint8)
            p.init_images[0] = Image.fromarray(self.parse_image(input_fg))

            if p.cfg_scale > 2.5:
                logger.warning("Low CFG is recommended!")
            if p.denoising_strength < 0.9:
                logger.warning("High Denoising Strength is recommended!")

        else:
            self.input_fg: np.ndarray = self.parse_image(input_fg)

        self.uploaded_bg: np.ndarray = self.parse_image(uploaded_bg)

        self.input_fg_rgb: np.ndarray = self.process_input_foreground(
            self.input_fg,
            remove_bg,
            rembg_model,
            foreground_threshold,
            background_threshold,
            erode_size,
        )

        self.detail_transfer = DetailTransfer(
            detail_transfer,
            detail_transfer_blur_radius,
            self.input_fg if detail_transfer_raw else self.input_fg_rgb,
        )

        if detail_transfer and reinforce_fg:
            assert isinstance(p, StableDiffusionProcessingImg2Img)
            assert self.model_type == ICModels.fc

            lightmap = np.asarray(p.init_images[0], dtype=np.uint8)

            mask = np.all(self.input_fg_rgb == np.asarray([127, 127, 127]), axis=-1)
            mask = mask[..., None]  # [H, W, 1]
            lightmap = resize_and_center_crop(
                lightmap,
                w=self.input_fg_rgb.shape[1],
                h=self.input_fg_rgb.shape[0],
            )
            lightmap_rgb = lightmap[..., :3]
            lightmap_alpha = lightmap[..., 3:4]
            lightmap_rgb = self.input_fg_rgb * (1 - mask) + lightmap_rgb * mask
            lightmap = np.concatenate([lightmap_rgb, lightmap_alpha], axis=-1)

            p.init_images[0] = Image.fromarray(lightmap.astype(np.uint8))

    @staticmethod
    def process_input_foreground(
        image: np.ndarray,
        remove_bg: bool,
        rembg_model: str,
        foreground_threshold: int,
        background_threshold: int,
        erode_size: int,
    ) -> np.ndarray:
        """Process input foreground image into [H, W, 3] format"""

        if image is None:
            return None

        if remove_bg:
            return run_rmbg(
                image,
                rembg_model,
                foreground_threshold,
                background_threshold,
                erode_size,
            )

        assert len(image.shape) == 3, "Does not support greyscale image..."

        if image.shape[2] == 3:
            return image

        return make_masked_area_grey(
            image[..., :3],
            image[..., 3:].astype(np.float32) / 255.0,
        )

    def get_concat_cond(
        self,
        processed_fg: np.ndarray,  # subject with background removed
        p: StableDiffusionProcessing,
    ) -> np.ndarray:
        """Returns concat condition in [B, H, W, C] format."""

        if getattr(p, "is_hr_pass", False):
            assert isinstance(p, StableDiffusionProcessingTxt2Img)
            if p.hr_resize_x == 0 and p.hr_resize_y == 0:
                hr_x = int(p.width * p.hr_scale)
                hr_y = int(p.height * p.hr_scale)
            else:
                hr_y, hr_x = p.hr_resize_y, p.hr_resize_x
            image_width = align_dim_latent(hr_x)
            image_height = align_dim_latent(hr_y)

        else:
            image_width = p.width
            image_height = p.height

        fg = resize_and_center_crop(processed_fg, image_width, image_height)

        match self.model_type:
            case ICModels.fc:
                np_concat = [fg]
            case ICModels.fbc:
                bg = resize_and_center_crop(
                    self.uploaded_bg,
                    image_width,
                    image_height,
                )
                np_concat = [fg, bg]
            case _:
                raise ValueError

        return np.stack(np_concat, axis=0)

    @staticmethod
    def decode_base64(base64string: str) -> np.ndarray:
        return np.asarray(decode_base64_to_image(base64string), dtype=np.uint8)

    @staticmethod
    def parse_image(value) -> np.ndarray:
        if isinstance(value, str):
            return ICLightArgs.decode_base64(value)
        assert isinstance(value, np.ndarray) or (value is None)
        return value
