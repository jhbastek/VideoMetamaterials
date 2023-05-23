<h1 align="center">Inverse-design of nonlinear mechanical metamaterials<br>via video denoising diffusion models</h1>
<div align="center">
  <span class="author-block">
    <a>Jan-Hendrik Bastek</a><sup>1</sup> and</span>
  <span class="author-block">
    <a>Dennis M. Kochmann</a><sup>1</sup></span>
</div>
<div align="center">
  <span class="author-block"><sup>1</sup>ETH Zurich</span>
</div>

$~$
<p align="center"><img src="pred_light.gif#gh-light-mode-only" width="550"\></p>
<p align="center"><img src="pred_dark.gif#gh-dark-mode-only" width="550"\></p>

## Introduction

We introduce a framework to create mechanical metamaterials with a given nonlinear stress-strain response via video denoising diffusion as described in [TBA]. The code is adapted from the video diffusion architecture proposed by [Phil Wang](https://github.com/lucidrains/imagen-pytorch) based on [Imagen Video](https://imagen.research.google/video/).

To reproduce the results from the publication, start by cloning this repository via
```
git clone https://github.com/jhbastek/VideoMetamaterials.git
``` 
We use the [Accelerate](https://huggingface.co/docs/accelerate/index) library to speed up training. Please first configure your setup via ```accelerate config``` before training the diffusion model via
```
accelerate launch main.py
```

For further information, please first refer to the [TBA], the Supporting Information [TBA] or reach out to [Jan-Hendrik Bastek](mailto:jbastek@ethz.ch).

## Dependencies

The framework was developed and tested on Python 3.11 using CUDA 12.0 and requires the following Python packages.
Package | Version (>=)
:-|:-
`pytorch`       | `2.0.1`
`einops`        | `0.6.1`
`einops-exts`   | `0.6.1`
`rotary_embedding_torch` | `0.2.3`
`accelerate`    | `0.19.0`
`imageio`       | `2.28.1`
`tqdm`          | `4.65.0`
`wandb` (optional)        | `0.15.2`

## Citation

If this code is useful for your research, please cite [TBA].

