import gradio as gr
import numpy as np
from lib_iclight import VERSION, i2i_fc, raw, removal, t2i_fbc, t2i_fc
from lib_iclight.backend import detect_backend
from lib_iclight.backgrounds import BackgroundFC
from lib_iclight.detail_utils import restore_detail
from lib_iclight.logging import logger
from lib_iclight.model_loader import ICModels
from lib_iclight.parameters import ICLightArgs
from lib_iclight.rembg_utils import get_models
from lib_iclight.settings import ic_settings

from modules import scripts
from modules.script_callbacks import on_ui_settings
from modules.shared import opts
from modules.ui_components import InputAccordion

backend_type, apply_ic_light = detect_backend()


class ICLightScript(scripts.Script):
    def __init__(self):
        ICModels.detect_models()
        self.args: ICLightArgs

    def title(self):
        return "IC Light"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img) -> list[gr.components.Component]:
        with InputAccordion(False, label=f"{self.title()} {VERSION}") as enable:
            with gr.Row():
                model_type = gr.Dropdown(
                    label="Mode",
                    choices=[ICModels.fc, ICModels.fbc],
                    value=ICModels.fc,
                    interactive=(not is_img2img),
                )
                desc = gr.Markdown(
                    value=(i2i_fc if is_img2img else t2i_fc),
                    elem_classes=["ic-light-desc"],
                )

            with gr.Column(variant="panel"):
                with gr.Row():
                    input_fg = gr.Image(
                        label=("Lighting Conditioning" if is_img2img else "Foreground"),
                        source="upload",
                        type="numpy",
                        height=480,
                        visible=True,
                        image_mode="RGBA",
                    )
                    uploaded_bg = gr.Image(
                        label="Background",
                        source="upload",
                        type="numpy",
                        height=480,
                        visible=False,
                        image_mode="RGB",
                    )

                def parse_resolution(img: np.ndarray | None) -> list[int, int]:
                    if img is None:
                        return [gr.skip(), gr.skip()]

                    h, w, _ = img.shape
                    while (w > 2048) or (h > 2048):
                        w /= 2
                        h /= 2

                    return [round(w / 64) * 64, round(h / 64) * 64]

                if not is_img2img:
                    _sync: bool = getattr(opts, "ic_sync_dim", True)
                    with gr.Row(variant="compact", elem_classes=["ic-light-btns"]):
                        sync = gr.Button("Sync Resolution", visible=_sync)
                        sync.click(
                            fn=parse_resolution,
                            inputs=[input_fg],
                            outputs=[self.txt2img_width, self.txt2img_height],
                            show_progress="hidden",
                        )

                        flip_bg = gr.Button("Flip Background", visible=False)

            _sources = [bg.value for bg in BackgroundFC]
            background_source = gr.Radio(
                label="Background Source",
                choices=_sources,
                value=_sources[-1],
                visible=is_img2img,
                type="value",
            )

            with InputAccordion(True, label="Background Removal") as remove_bg:
                gr.Markdown(removal)

                _rembg_models = get_models()
                rembg_model = gr.Dropdown(
                    label="Background Removal Model",
                    choices=_rembg_models,
                    value=_rembg_models[0],
                )
                with gr.Row():
                    foreground_threshold = gr.Slider(
                        label="Foreground Threshold",
                        value=225,
                        minimum=0,
                        maximum=255,
                        step=1,
                    )
                    background_threshold = gr.Slider(
                        label="Background Threshold",
                        value=16,
                        minimum=0,
                        maximum=255,
                        step=1,
                    )
                erode_size = gr.Slider(
                    label="Erode Size",
                    value=16,
                    minimum=0,
                    maximum=128,
                    step=1,
                )

            with InputAccordion(False, label="Restore Details") as detail_transfer:
                detail_transfer_raw = gr.Checkbox(False, label=raw)
                detail_transfer_blur_radius = gr.Slider(
                    label="Blur Radius",
                    info="for Difference of Gaussian; higher = stronger",
                    value=3,
                    minimum=1,
                    maximum=9,
                    step=2,
                )

            with gr.Row(variant="compact", visible=is_img2img):
                reinforce_fg = gr.Checkbox(
                    value=False,
                    label="Reinforce Foreground",
                    info="Paste the Subject onto the Lighting Conditioning",
                )

        if is_img2img:
            self._hook_i2i(input_fg, background_source)
        else:
            self._hook_t2i(model_type, flip_bg, uploaded_bg, desc)

        components: list[gr.components.Component] = [
            enable,
            model_type,
            input_fg,
            uploaded_bg,
            remove_bg,
            rembg_model,
            foreground_threshold,
            background_threshold,
            erode_size,
            detail_transfer,
            detail_transfer_raw,
            detail_transfer_blur_radius,
            reinforce_fg,
        ]

        for comp in components:
            comp.do_not_save_to_config = True

        return components

    def before_process(self, p, enable: bool, *args, **kwargs):
        self.detailed_images: list = []
        self.args = None

        if not enable:
            return
        if not p.sd_model.is_sd1:
            logger.error("IC-Light only supports SD1 checkpoint...")
            return
        if args[1] is None:
            logger.error("An input image is required...")
            return

        self.args = ICLightArgs(p, *args)

    def process_before_every_sampling(self, p, *args, **kwargs):
        if self.args is not None:
            apply_ic_light(p, self.args)

    def postprocess_image(self, p, pp, *args, **kwargs):
        if self.args is None:
            return
        if not self.args.detail_transfer.enable:
            return

        self.detailed_images.append(
            restore_detail(
                np.asarray(pp.image).astype(np.uint8),
                self.args.detail_transfer.original,
                self.args.detail_transfer.radius,
            )
        )

    def postprocess(self, p, processed, *args, **kwargs):
        if self.args is None:
            return

        processed.images.extend(self.detailed_images)

    def after_component(self, component: gr.Slider, **kwargs):
        if not getattr(opts, "ic_sync_dim", True):
            return

        if not (elem_id := kwargs.get("elem_id", None)):
            return

        if elem_id == "txt2img_width":
            self.txt2img_width = component
        if elem_id == "txt2img_height":
            self.txt2img_height = component

    @staticmethod
    def _hook_t2i(model_type: gr.Dropdown, flip_bg: gr.Button, uploaded_bg, desc):
        def on_model_change(model: str):
            match model:
                case ICModels.fc:
                    return (
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(value=t2i_fc),
                    )
                case ICModels.fbc:
                    return (
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(value=t2i_fbc),
                    )
                case _:
                    raise ValueError

        model_type.change(
            fn=on_model_change,
            inputs=[model_type],
            outputs=[flip_bg, uploaded_bg, desc],
            show_progress="hidden",
        )

        def on_flip_image(image: np.ndarray) -> np.ndarray:
            if image is None:
                return gr.skip()
            return gr.update(value=np.fliplr(image))

        flip_bg.click(fn=on_flip_image, inputs=[uploaded_bg], outputs=[uploaded_bg])

    @staticmethod
    def _hook_i2i(input_fg: gr.Image, background_source: gr.Dropdown):
        def update_img2img_input(source: str):
            source_fc = BackgroundFC(source)
            if source_fc is BackgroundFC.CUSTOM:
                return gr.skip()
            else:
                return gr.update(value=source_fc.get_bg())

        background_source.input(
            fn=update_img2img_input,
            inputs=[background_source],
            outputs=[input_fg],
            show_progress="hidden",
        )

        input_fg.upload(
            fn=lambda: gr.update(value=BackgroundFC.CUSTOM.value),
            outputs=[background_source],
            show_progress="hidden",
        )


on_ui_settings(ic_settings)
