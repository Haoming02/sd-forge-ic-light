from modules import scripts, script_callbacks
from modules.ui_components import InputAccordion
from modules.paths import models_path
from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
)

from lib_iclight.args import ICLightArgs, BGSourceFC, BGSourceFBC
from lib_iclight.model_loader import ModelType, detect_models
from lib_iclight.ic_modes import t2i_fc, t2i_fbc, i2i_fc
from lib_iclight.rembg_utils import AVAILABLE_MODELS
from lib_iclight.detail_utils import restore_detail

from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import gradio as gr
import numpy as np


class BackendType(Enum):
    A1111 = "A1111"
    Forge = "Forge"


@dataclass
class A1111Context:
    """Contains all components from A1111."""

    txt2img_submit_button: Optional[gr.components.Component] = None
    img2img_submit_button: Optional[gr.components.Component] = None

    # Slider controls from A1111 WebUI.
    img2img_w_slider: Optional[gr.components.Component] = None
    img2img_h_slider: Optional[gr.components.Component] = None

    img2img_image: Optional[gr.Image] = None

    def set_component(self, component: gr.components.Component):
        id_mapping = {
            "txt2img_generate": "txt2img_submit_button",
            "img2img_generate": "img2img_submit_button",
            "img2img_width": "img2img_w_slider",
            "img2img_height": "img2img_h_slider",
            "img2img_image": "img2img_image",
        }
        elem_id = getattr(component, "elem_id", None)
        if elem_id in id_mapping and getattr(self, id_mapping[elem_id]) is None:
            setattr(self, id_mapping[elem_id], component)


class ICLightScript(scripts.Script):
    DEFAULT_ARGS = ICLightArgs()
    a1111_context = A1111Context()

    def __init__(self) -> None:
        super().__init__()
        self.args: ICLightArgs = None

        try:
            from lib_iclight.forge_backend import apply_ic_light

            self.apply_ic_light = apply_ic_light
            self.backend_type = BackendType.Forge

        except ImportError:
            from lib_iclight.a1111_backend import apply_ic_light

            self.apply_ic_light = apply_ic_light
            self.backend_type = BackendType.A1111

    def title(self):
        return "IC Light"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool) -> Tuple[gr.components.Component, ...]:

        if is_img2img:
            model_type_choices = [ModelType.FC.name]
            bg_source_fc_choices = [e.value for e in BGSourceFC if e != BGSourceFC.NONE]
        else:
            model_type_choices = [ModelType.FC.name, ModelType.FBC.name]
            bg_source_fc_choices = [BGSourceFC.NONE.value]

        with InputAccordion(value=False, label=self.title()) as enabled:
            with gr.Row():
                model_type = gr.Dropdown(
                    label="Mode",
                    choices=model_type_choices,
                    value=ModelType.FC.name,
                    interactive=(not is_img2img),
                )

                desc = gr.Markdown(
                    value=i2i_fc if is_img2img else t2i_fc,
                    elem_classes=["ic-light-desc"],
                )

            with gr.Row():
                input_fg = gr.Image(
                    source="upload",
                    type="numpy",
                    label=("Lighting Conditioning" if is_img2img else "Foreground"),
                    height=480,
                    interactive=True,
                    visible=True,
                    image_mode="RGBA",
                )
                uploaded_bg = gr.Image(
                    source="upload",
                    type="numpy",
                    label="Background",
                    height=480,
                    interactive=True,
                    visible=False,
                )

            reinforce_fg = gr.Checkbox(
                label="Reinforce Foreground",
                info="Preserve the foreground base color",
                value=False,
                interactive=True,
                visible=is_img2img,
            )

            with InputAccordion(value=False, label="Background Removal") as remove_bg:
                gr.Markdown(
                    "<i>Disable if you already have a subject with the background removed</i>"
                )

                rembg_model = gr.Dropdown(
                    label="Background Removal Model",
                    choices=AVAILABLE_MODELS,
                    value=AVAILABLE_MODELS[0],
                )

                foreground_threshold = gr.Slider(
                    label="Foreground Threshold",
                    minimum=0,
                    maximum=255,
                    step=1,
                    value=225,
                )

                background_threshold = gr.Slider(
                    label="Background Threshold",
                    minimum=0,
                    maximum=255,
                    step=1,
                    value=16,
                )

                erode_size = gr.Slider(
                    label="Erode Size",
                    minimum=0,
                    maximum=128,
                    step=1,
                    value=16,
                )

            bg_source_fc = gr.Radio(
                label="Background Source",
                choices=bg_source_fc_choices,
                value=(
                    BGSourceFC.CUSTOM.value if is_img2img else BGSourceFC.NONE.value
                ),
                type="value",
                visible=is_img2img,
                interactive=True,
            )

            bg_source_fbc = gr.Radio(
                label="Background Source",
                choices=[BGSourceFBC.UPLOAD.value, BGSourceFBC.UPLOAD_FLIP.value],
                value=BGSourceFBC.UPLOAD.value,
                type="value",
                visible=False,
                interactive=True,
            )

            with InputAccordion(
                value=False, label="Restore Details"
            ) as detail_transfer:

                detail_transfer_use_raw_input = gr.Checkbox(
                    label="Use the [Original Input] instead of the [Image with Background Removed]"
                )

                detail_transfer_blur_radius = gr.Slider(
                    label="Blur Radius",
                    info="for Difference of Gaussian",
                    minimum=1,
                    maximum=9,
                    step=2,
                    value=5,
                )

        state = gr.State({})
        (
            ICLightScript.a1111_context.img2img_submit_button
            if is_img2img
            else ICLightScript.a1111_context.txt2img_submit_button
        ).click(
            fn=lambda *args: dict(
                zip(
                    vars(self.DEFAULT_ARGS).keys(),
                    args,
                )
            ),
            inputs=[
                enabled,
                model_type,
                input_fg,
                uploaded_bg,
                bg_source_fc,
                bg_source_fbc,
                remove_bg,
                rembg_model,
                foreground_threshold,
                background_threshold,
                erode_size,
                reinforce_fg,
                detail_transfer,
                detail_transfer_use_raw_input,
                detail_transfer_blur_radius,
            ],
            outputs=state,
            queue=False,
        )

        if is_img2img:

            def update_img2img_input(bg_source_fc: str):
                bg_source_fc = BGSourceFC(bg_source_fc)
                if bg_source_fc == BGSourceFC.CUSTOM:
                    return gr.skip()

                return gr.update(value=bg_source_fc.get_bg(512, 512))

            bg_source_fc.input(
                fn=update_img2img_input,
                inputs=[bg_source_fc],
                outputs=[input_fg],
            )

            def set_img2img_mode():
                return gr.update(value=BGSourceFC.CUSTOM)

            input_fg.upload(
                fn=set_img2img_mode,
                inputs=None,
                outputs=[bg_source_fc],
                show_progress="hidden",
            )

        else:

            def on_model_change(model_type: str):
                match ModelType.get(model_type):
                    case ModelType.FC:
                        return (
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(value=t2i_fc),
                        )
                    case ModelType.FBC:
                        return (
                            gr.update(visible=True),
                            gr.update(visible=True),
                            gr.update(value=t2i_fbc),
                        )
                    case _:
                        raise SystemError

            model_type.change(
                fn=on_model_change,
                inputs=[model_type],
                outputs=[bg_source_fbc, uploaded_bg, desc],
                show_progress=False,
            )

        return (state,)

    def before_process(self, p, *args, **kwargs):
        self.detailed_images: list = []

        _args = ICLightArgs.fetch_from(p)
        if not _args:
            self.args = None
            return

        if isinstance(p, StableDiffusionProcessingImg2Img):
            p.init_images[0] = _args.get_lightmap(p)

        self.args = _args

    def process_before_every_sampling(
        self, p: StableDiffusionProcessing, *args, **kwargs
    ):
        """
        Similar to process(), called before every sampling.
        If you use high-res fix, this will be called twice.
        """
        if (self.args is None) or (not self.args.enabled):
            return

        self.apply_ic_light(p, self.args)

    def postprocess_image(self, p, pp, *args, **kwargs):
        if (
            (self.args is None)
            or (not self.args.enabled)
            or (not self.args.detail_transfer)
        ):
            return

        self.detailed_images.append(
            restore_detail(
                np.asarray(pp.image).astype(np.uint8),
                (
                    self.args.input_fg
                    if self.args.detail_transfer_use_raw_input
                    else self.args.input_fg_rgb
                ),
                int(self.args.detail_transfer_blur_radius),
            )
        )

    def postprocess(self, p, processed, *args, **kwargs):
        if (self.args is None) or (not self.args.enabled):
            return
        if self.backend_type == BackendType.A1111:
            if getattr(p, "extra_result_images", None):
                processed.images += p.extra_result_images
        if self.detailed_images:
            processed.images += self.detailed_images

    @staticmethod
    def on_after_component(component, **_kwargs):
        """Register the A1111 component."""
        ICLightScript.a1111_context.set_component(component)


script_callbacks.on_after_component(ICLightScript.on_after_component)
script_callbacks.on_before_ui(lambda: detect_models(models_path))
