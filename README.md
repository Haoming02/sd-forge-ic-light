# SD Forge IC-Light
This is an Extension for the [Forge Webui](https://github.com/lllyasviel/stable-diffusion-webui-forge), which implements [IC-Light](https://github.com/lllyasviel/IC-Light), allowing you to manipulate the illumination of images.

> This only works with **SD 1.5** checkpoints

<details>
<summary>for <b>Automatic1111 Webui</b></summary>

- Only version **v1.10.0** or later is supported
- You also need to install [sd-webui-model-patcher](https://github.com/huchenlei/sd-webui-model-patcher) first

</details>

## Getting Started
1. Download the <ins>two</ins> models from [Releases](https://github.com/Haoming02/sd-forge-ic-light/releases)
2. Create a new folder, `ic-light`, inside your webui `models` folder
3. Place the 2 models inside said folder
4. **(Optional)** You can rename the models, as long as the filenames contain either **`fc`** or **`fbc`**

## How to use
> Recommended to use low CFG and strong denosing strength

**W.I.P**

## Roadmap
- [ ] How to Use
- [X] Select different `rembg` models
- [ ] API Support
- [ ] Improve DoG

## Known Issue
- [ ] If you click `Reuse Seed` when previewing the appended images instead of the first result image, it will result in an error.
    > This is mostly upstream, as even the built-in ControlNet raises this error. I probably won't address it until the Webui has an unified way to append images...

<hr>

> [!NOTE]
> This fork has been heavily rewritten. I will still try to merge any backend changes upstream; however, the frontend will retain my opinionated breaking changes. Therefore, merging this fork is highly discouraged without thorough testing.
