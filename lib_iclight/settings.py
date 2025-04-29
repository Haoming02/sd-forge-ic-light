from modules.shared import OptionInfo, opts


def ic_settings():
    args = {"section": ("ic", "IC Light"), "category_id": "sd"}

    opts.add_option(
        "ic_sync_dim",
        OptionInfo(True, "Show [Sync Resolution] Button", **args).needs_reload_ui(),
    )

    opts.add_option(
        "ic_all_rembg",
        OptionInfo(False, "List all available Rembg models", **args).needs_reload_ui(),
    )
