$~$

<p align="center"><img src="pred_light.gif#gh-light-mode-only" width="550"\></p>
<p align="center"><img src="pred_dark.gif#gh-dark-mode-only" width="550"\></p>

# Inverse-design of nonlinear mechanical metamaterials via video denoising diffusion models

We introduce a framework to create mechanical metamaterials with a given nonlinear stress-strain response via video denoising diffusion as described in [TBA].

To run the studies, simply clone this repository via
```
git clone https://github.com/jhbastek/VideoMetamaterials.git
```
and run **main.py** with the indicated study.

For further information, please first refer to the [TBA], the Supporting Information [TBA] or reach out to [Jan-Hendrik Bastek](mailto:jbastek@ethz.ch).

## Dependencies

The framework was developed and tested on Python 3.11 using CUDA 12.0 and relies on the following Python packages.
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

