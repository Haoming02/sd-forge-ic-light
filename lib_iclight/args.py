from modules.processing import (
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
    StableDiffusionProcessing,
)

from modules.api.api import decode_base64_to_image

from .bg_source import BGSourceFC, BGSourceFBC
from .model_loader import ModelType
from .rembg_utils import run_rmbg
from .utils import (
    resize_and_center_crop,
    make_masked_area_grey,
    align_dim_latent,
)

from PIL import Image
import numpy as np


class ICLightArgs:

    @staticmethod
    def parse_model_type(value) -> ModelType:
        if isinstance(value, str):
            return ModelType.get(value)
        assert isinstance(value, ModelType)
        return value

    @staticmethod
    def cls_decode_base64(base64string: str) -> np.ndarray:
        return np.array(decode_base64_to_image(base64string)).astype("uint8")

    @staticmethod
    def parse_image(value) -> np.ndarray:
        if isinstance(value, str):
            return ICLightArgs.cls_decode_base64(value)
        assert isinstance(value, np.ndarray) or (value is None)
        return value

    def __init__(self, p: StableDiffusionProcessing, args: tuple):
        self.enabled: bool = args[0]

        self.model_type: ModelType = self.parse_model_type(args[1])
        self.bg_source_fc: BGSourceFC = BGSourceFC(args[2])
        self.bg_source_fbc: BGSourceFBC = BGSourceFBC(args[3])

        if isinstance(p, StableDiffusionProcessingImg2Img):
            input_image = np.asarray(p.init_images[0]).astype(np.uint8)
            p.init_images[0] = Image.fromarray(args[4])
            self.input_fg: np.ndarray = input_image
        else:
            self.input_fg: np.ndarray = self.parse_image(args[4])

        self.uploaded_bg: np.ndarray = self.parse_image(args[5])

        self.remove_bg: bool = args[6]
        self.rembg_model: str = args[7]
        self.foreground_threshold = int(args[8])
        self.background_threshold = int(args[9])
        self.erode_size = int(args[10])

        self.detail_transfer: bool = args[11]
        self.detail_transfer_use_raw_input: bool = args[12] or (not self.remove_bg)
        self.detail_transfer_blur_radius = int(args[13])

        self.input_fg_rgb: np.ndarray = self.process_input_fg()

        self.reinforce_fg: bool = args[14]

        if self.reinforce_fg:
            assert self.model_type == ModelType.FC
            assert isinstance(p, StableDiffusionProcessingImg2Img)

            lightmap = np.asarray(p.init_images[0]).astype(np.uint8)

            mask = np.all(self.input_fg_rgb == np.array([127, 127, 127]), axis=-1)
            mask = mask[..., None]  # [H, W, 1]
            lightmap = resize_and_center_crop(
                lightmap,
                target_width=self.input_fg_rgb.shape[1],
                target_height=self.input_fg_rgb.shape[0],
            )
            lightmap_rgb = lightmap[..., :3]
            lightmap_alpha = lightmap[..., 3:4]
            lightmap_rgb = self.input_fg_rgb * (1 - mask) + lightmap_rgb * mask
            lightmap = np.concatenate([lightmap_rgb, lightmap_alpha], axis=-1).astype(
                np.uint8
            )

            p.init_images[0] = Image.fromarray(lightmap)

    def process_input_fg(self) -> np.ndarray:
        """Process input fg image into format [H, W, C=3]"""

        rgb: np.ndarray = None
        if self.input_fg is None:
            return rgb

        if self.remove_bg:
            rgb = run_rmbg(
                self.input_fg,
                self.rembg_model,
                self.foreground_threshold,
                self.background_threshold,
                self.erode_size,
            )

        else:
            if len(self.input_fg.shape) < 3:
                raise NotImplementedError("Does not support greyscale image...")

            if self.input_fg.shape[-1] == 4:
                rgb = make_masked_area_grey(
                    self.input_fg[..., :3],
                    self.input_fg[..., 3:].astype(np.float32) / 255.0,
                )
            else:
                rgb = self.input_fg

        if rgb.shape[2] != 3:
            raise ValueError("Input Image should be in RGB channels...")

        return rgb

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
            case ModelType.FC:
                np_concat = [fg]
            case ModelType.FBC:
                bg = resize_and_center_crop(
                    self.bg_source_fbc.get_bg(
                        image_width, image_height, self.uploaded_bg
                    ),
                    image_width,
                    image_height,
                )
                np_concat = [fg, bg]
            case _:
                raise SystemError

        return np.stack(np_concat, axis=0)
