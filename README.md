# SD Forge IC-Light
This is an Extension for the [Forge Webui](https://github.com/lllyasviel/stable-diffusion-webui-forge), which implements [IC-Light](https://github.com/lllyasviel/IC-Light), allowing you to manipulate the illumination of images.

### Compatibility Matrix

> **Last Checked:** 2025 Apr.29

<table>
    <tr align="center">
        <th>Automatic1111<br><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/tree/dev">dev</a></th>
        <th><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">Forge</a><br>(Gradio 4)</th>
        <th>Forge <a href="https://github.com/Haoming02/sd-webui-forge-classic">Classic</a><br>(Gradio 3)</th>
        <th>reForge<br><a href="https://github.com/Panchovix/stable-diffusion-webui-reForge/tree/main">main</a></th>
        <th>reForge<br><a href="https://github.com/Panchovix/stable-diffusion-webui-reForge/tree/dev">dev</a></th>
        <th>reForge<br><a href="https://github.com/Panchovix/stable-diffusion-webui-reForge/tree/dev2">dev2</a></th>
    </tr>
    <tr align="center">
        <td><b>Working</b></td>
        <td><b>Working</b></td>
        <td><b>Working</b></td>
        <td><b>Working</b></td>
        <td><b>Working</b></td>
        <td><b>Working</b></td>
    </tr>
</table>

<details>
<summary>for <b>Automatic1111 Webui</b></summary>

- You additionally need to install [sd-webui-model-patcher](https://github.com/huchenlei/sd-webui-model-patcher) first

</details>

## Getting Started
1. Download the <ins><b>two</b></ins> models from [Releases](https://github.com/Haoming02/sd-forge-ic-light/releases)
2. Create a new folder, `ic-light`, inside your webui `models` folder
3. Place the two models inside the `ic-light` folder
4. **(Optional)** You can rename the models, as long as the filename contains either **`fc`** or **`fbc`**

## How to Use

> [!Important]
> IC-Light only supports **SD1** checkpoints

#### Index

1. [txt2img - FC](#txt2img---fc)
2. [txt2img - FBC](#txt2img---fbc)
3. [img2img - FC](#img2img---fc)
    - [Reinforce Foreground](#reinforce-foreground)
4. Options
    - [Background Removal](#background-removal)
    - [Restore Details](#restore-details)

<p align="center">
<img src="assets/subject_input.jpg" width=256><br>
example <code>Foreground</code> image
</p>

### txt2img - FC
> Relighting with Foreground Condition

- In the Extension input, upload an image of your subject, then generate a new background using **txt2img**
- If the generation aspect ratio is different, the `Foreground` image will be `Crop and resize` first
- `Hires. Fix` is supported

<p align="center">
<img src="assets/fc_output.jpg" width=384><br>
example output<br>
<code>a photo of a gentleman in suit, standing under sunset</code>
</p>

### txt2img - FBC
> Relighting with Foreground and Background Condition

- In the Extension inputs, upload an image of your subject, and another image as the background
- Simply write some quality tags as the prompts
- `Hires. Fix` is supported

<p align="center">
<img src="assets/bg_beach.jpg" width=384><br>
example <code>Background</code> image
</p>

<p align="center">
<img src="assets/fbc_output.jpg" width=384><br>
example output<br>
<code>a photo of a gentleman in suit, standing at a beach, sunny day</code>
</p>

### img2img - FC
> Relighting with Light-Map Condition

- In the **img2img** input, upload an image of your subject as normal
- In the Extension input, you can select between different light directions, or select `Custom LightMap` and upload one yourself
- Describe the scene with the prompts
- Low `CFG` *(`~2.0`)* and high `Denoising strength` *(`~ 1.0`)* is recommended

<p align="center">
<img src="assets/i2i_output1.jpg" width=384>
<img src="assets/i2i_output2.jpg" width=384><br>
example output<br>
<code>Right Light</code> | <code>Left Light</code><br>
<code>a photo of a gentleman in suit, winter, snowing</code>
</p>

#### Reinforce Foreground

When enabled, the subject will be additionally pasted onto the light map to preserve the original color. This may improve the details at the cost of weaker lighting influence.

> As shown below, the suit gets brightened to a khaki color by the prompt; using `Reinforce Foreground` allows the suit to retain more of its original color

<p align="center">
<img src="assets/reinforce_off.jpg" width=256>
<img src="assets/reinforce_on.jpg" width=256><br>
example output<br>
<code>Off | On</code><br>
<code>fire, explosion</code>
</p>

<br>

### Options
> These settings are available for all 3 modes

#### Background Removal

- Use the **[rembg](https://github.com/danielgatis/rembg)** package to separate the subject from the background.
- If you already have a subject image with alpha, you can simply disable this option.
- If you have an anime subject instead, select `isnet-anime` from the **Background Removal Model** dropdown.
- When this is enabled, it will additionally append the result to the outputs.
- If the separation is not clean enough, edit the **Threshold** parameters to improve the accuracy.

<p align="center">
<img src="assets/subject_rembg.jpg" width=256><br>
example result
</p>

#### Restore Details

Use the *Difference of Gaussian* algorithm to transfer the details from the input to the output.

By default, this only uses the `DoG` of the subject without background. You can also switch to using the `DoG` of the entire input image instead. Increasing the **Blur Radius** will strengthen the effect.

<br>

## Settings

> The settings are in the **IC Light** section under the <ins>Stable Diffusion</ins> category in the **Settings** tab

- **Sync Resolution Button:** Adds a button in the `txt2img` tab that changes the `Width` and `Height` parameters to the closest ratio of the uploaded `Foreground` image.
- **All Rembg Models:** By default, the Extension only shows `u2net_human_seg` and `isnet-anime` options. If those do not suit your needs *(**eg.** your subject is not a "person")*, you may enable this to list all available models instead.

## Roadmap
- [X] Select different `rembg` models
- [X] API Support
    - see [wiki](https://github.com/Haoming02/sd-forge-ic-light/wiki/API)
- [ ] Improve `Reinforce Foreground`
- [ ] Improve `Restore Details`

<hr>

<pre align="center">
Copyright 2024 huchenlei
Copyright 2025 Haoming02

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
</pre>
