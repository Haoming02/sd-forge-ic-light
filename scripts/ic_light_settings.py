from modules.script_callbacks import on_ui_settings
from modules.shared import OptionInfo, opts

section = ("ic", "IC Light")


def settings():
    opts.add_option(
        "ic_sync_dim",
        OptionInfo(
            True,
            "Show [Sync Resolution] button in txt2img",
            section=section,
            category_id="sd",
        ).needs_reload_ui(),
    )

    opts.add_option(
        "ic_all_rembg",
        OptionInfo(
            False,
            "List all available Rembg models",
            section=section,
            category_id="sd",
        ).needs_reload_ui(),
    )


on_ui_settings(settings)
