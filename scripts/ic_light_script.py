from modules.processing import StableDiffusionProcessingImg2Img
from modules.ui_components import InputAccordion
from modules import scripts, script_callbacks

from lib_iclight.model_loader import ModelType, detect_models
from lib_iclight.bg_source import BGSourceFC, BGSourceFBC
from lib_iclight.ic_modes import t2i_fc, t2i_fbc, i2i_fc
from lib_iclight.rembg_utils import AVAILABLE_MODELS
from lib_iclight.detail_utils import restore_detail
from lib_iclight.args import ICLightArgs

from enum import Enum
import gradio as gr
import numpy as np


class BackendType(Enum):
    A1111 = "A1111"
    Forge = "Forge"


class ICLightScript(scripts.Script):

    def __init__(self):
        self.args: ICLightArgs = None

        try:
            from lib_iclight.forge_backend import apply_ic_light

            self.apply_ic_light = apply_ic_light
            self.backend_type = BackendType.Forge

        except ImportError:
            from lib_iclight.a1111_backend import apply_ic_light

            self.apply_ic_light = apply_ic_light
            self.backend_type = BackendType.A1111

            from modules.launch_utils import git_tag

            version = git_tag()
            if version == "<none>":
                return

            major, minor, rev = version.split(".", 2)
            if int(minor) < 10:
                raise NotImplementedError(
                    "\n[IC-Light] Only Automatic1111 v1.10.0 or later is supported!\n"
                )

    def title(self):
        return "IC Light"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool) -> list[gr.components.Component]:

        bg_source_fc_choices = (
            [e.value for e in BGSourceFC if e != BGSourceFC.NONE]
            if is_img2img
            else [BGSourceFC.NONE.value]
        )

        with InputAccordion(value=False, label=self.title()) as enabled:
            with gr.Row():
                model_type = gr.Dropdown(
                    label="Mode",
                    choices=(
                        [ModelType.FC.name, ModelType.FBC.name]
                        if (not is_img2img)
                        else [ModelType.FC.name]
                    ),
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

            bg_source_fc = gr.Radio(
                label="Background Source",
                choices=bg_source_fc_choices,
                value=bg_source_fc_choices[-1],
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

            with InputAccordion(value=True, label="Background Removal") as remove_bg:
                gr.Markdown("<i>Disable if the subject already has no background</i>")

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

            with InputAccordion(
                value=False, label="Restore Details"
            ) as detail_transfer:

                detail_transfer_use_raw_input = gr.Checkbox(
                    label="Use the [Original Input] instead of the [Subject with Background Removed]"
                )

                detail_transfer_blur_radius = gr.Slider(
                    label="Blur Radius",
                    info="for Difference of Gaussian",
                    minimum=1,
                    maximum=9,
                    step=2,
                    value=3,
                )

            reinforce_fg = gr.Checkbox(
                label="Reinforce Foreground",
                info="Paste the Subject onto the Lighting Conditioning",
                value=False,
                interactive=True,
                visible=is_img2img,
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

        components: list = [
            enabled,
            model_type,
            bg_source_fc,
            bg_source_fbc,
            input_fg,
            uploaded_bg,
            remove_bg,
            rembg_model,
            foreground_threshold,
            background_threshold,
            erode_size,
            detail_transfer,
            detail_transfer_use_raw_input,
            detail_transfer_blur_radius,
            reinforce_fg,
        ]

        for comp in components:
            comp.do_not_save_to_config = True

        return components

    def before_process(self, p, *args):
        self.detailed_images: list = []

        if not bool(args[0]):
            self.args = None
        else:
            self.args = ICLightArgs(p, args)
            p.extra_generation_params["IC-Light"] = True

    def process_before_every_sampling(self, p, *args, **kwargs):
        if not (self.args and getattr(self.args, "enabled", False)):
            return

        self.apply_ic_light(p, self.args)

    def postprocess_image(self, p, pp, *args, **kwargs):
        if not (
            self.args
            and getattr(self.args, "enabled", False)
            and getattr(self.args, "detail_transfer", False)
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
                self.args.detail_transfer_blur_radius,
            )
        )

    def postprocess(self, p, processed, *args, **kwargs):
        if not (self.args and getattr(self.args, "enabled", False)):
            return

        if self.backend_type == BackendType.A1111:
            if extras := getattr(p, "extra_result_images", None):
                processed.images += extras

        if self.detailed_images:
            processed.images += self.detailed_images


script_callbacks.on_before_ui(detect_models)
