<h1 align="center">Inverse-design of nonlinear mechanical metamaterials<br>via video denoising diffusion models</h1>
<h4 align="center">
<a href="https://arxiv.org/abs/2305.19836"><img src="https://img.shields.io/badge/arXiv-2305.19836-blue" alt="arXiv"></a>
<a href="https://doi.org/10.5281/zenodo.10011767"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.10011767.svg" alt="DOI"></a>  
</h4>
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

## Introduction & Setup
We introduce a framework to create mechanical metamaterials with a given nonlinear stress-strain response via video denoising diffusion as described in [Inverse design of nonlinear mechanical metamaterials via video denoising diffusion models](https://www.nature.com/articles/s42256-023-00762-x).

This code is based on the video denoising diffusion implementation by [Phil Wang](https://github.com/lucidrains/imagen-pytorch) proposed in [Imagen Video](https://imagen.research.google/video/).

To conduct similar studies as those presented in the publication, start by cloning this repository via
```
git clone https://github.com/jhbastek/VideoMetamaterials.git
```

Next, download the data and model checkpoints provided in the [ETHZ Research Collection](https://doi.org/10.3929/ethz-b-000629716). Unzip the training data `lagrangian.zip` in the `data` folder and the pre-trained model `pretrained.zip` in the `runs` folder, as shown below. Note that `eulerian.zip` must only be provided when training the model in the Eulerian frame, which was only used in preliminary studies.
```
.
├── data
│   ├── target_responses.csv
│   └── lagrangian
│   │   └── ...
│   └── eulerian (optional)
│       └── ...
└── runs
    └── pretrained
        └── ...
```

We use the [Accelerate](https://huggingface.co/docs/accelerate/index) library to speed up training when a multi GPU environment is available. Please first configure your setup via `accelerate config` (note that `accelerate` can also be used in single GPU/CPU setups).

To generate new metamaterial samples conditioned on the four stress-strain responses shown in the publication simply run
```
accelerate launch main.py
```
The generated samples will then be stored in `runs/pretrained/eval_target_w_<guidance_weight>/` and should perform similar to the presented samples. We arrange all generated samples in a single grid, in which the row corresponds to row of `data/target_responses.csv`.   

To condition the denoising process on your own stress-strain responses, simply adjust `data/target_responses.csv` accordingly. Sample generation takes around 1 minute on a single Nvidia Quadro RTX 6000. In case of interest, we store the normalization constants to rescale the pixel values to their physical equivalent in `data/<reference_frame>/training/min_max_values.csv`.

To experiment with different setups simply change the user input in `main.py`. Here you can adjust the number of generated samples per conditioning, change the guidance scaling `w` or also train denoising models from scratch based on the hyperparameters defined in `model.yaml` (including the option to log to [Weights & Biases](https://wandb.ai)).

For further information, please first refer to the [publication](https://www.nature.com/articles/s42256-023-00762-x), the [Supplementary Information](https://www.nature.com/articles/s42256-023-00762-x#Sec18) or reach out to [Jan-Hendrik Bastek](mailto:jbastek@ethz.ch).

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

If this code is useful for your research, please cite our [publication](https://www.nature.com/articles/s42256-023-00762-x).
```bibtex
@article{Bastek2023,
author = {Bastek, Jan-Hendrik and Kochmann, Dennis M.},
doi = {10.1038/s42256-023-00762-x},
journal = {Nature Machine Intelligence},
pages = {104849},
title = {{Inverse design of nonlinear mechanical metamaterials via video denoising diffusion models}},
url = {https://doi.org/10.1038/s42256-023-00762-x},
volume = {12},
year = {2023}
}

