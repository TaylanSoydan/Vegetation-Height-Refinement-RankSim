# Vegetation Height Refinement with RankSim

Self-supervised refinement of coarse LiDAR-derived vegetation height maps (VHM) using Sentinel-2 satellite imagery as guidance, enhanced with RankSim ranking-similarity loss for imbalanced height distributions.

**Report:** [Refinement of Large-scale Vegetation Height Maps](Refinement_of_Large_scale_Vegetation_Height_Maps.pdf)

## Overview

Global vegetation height maps derived from spaceborne LiDAR (GEDI) fused with Sentinel-2 satellite imagery achieve 10m resolution, but the coarse GEDI footprint (~25m) causes an effective loss of spatial detail. This project refines those coarse maps by formulating the problem as a self-supervised guided super-resolution task — no fine-resolution ground truth is used during training.

The inputs are:
- **Source** — coarse VHM derived from GEDI LiDAR + Sentinel-2 fusion
- **Guide** — high-resolution Sentinel-2 satellite RGB imagery (10m)
- **Segmentation** — Dynamic World land cover classes from Google Earth Engine

Fine airborne laser scanner (ALS) LiDAR data serves as ground truth for evaluation only.

### Key Contribution

Vegetation height distributions are heavily imbalanced — most pixels are bare ground or low vegetation, while tall canopy is rare. We adapt [RankSim](https://github.com/BorealisAI/ranksim-imbalanced-regression), a ranking-similarity regularizer for imbalanced regression, to the [PixTransform](https://github.com/prs-eth/PixTransform) super-resolution architecture. RankSim encourages the model's internal feature rankings to preserve the relative ordering of vegetation heights across samples, improving reconstruction quality for underrepresented tall-canopy regions.

## Architecture

The model uses a dual-branch PixTransform design:

| Branch | Input | Role |
|---|---|---|
| **Spatial** | 2D coordinate grid (x, y) | Learns position-dependent height patterns |
| **Color** | Sentinel-2 RGB channels | Learns spectral-to-height mappings |

Both branches produce 2048-dim feature maps that are summed and passed through a shared head network to predict per-pixel vegetation height. An optional positional encoding variant replaces the spatial branch with sinusoidal embeddings.

**Training:** The model is trained per-scene — the only supervision is that the mean predicted height over each coarse pixel must match the input coarse VHM value (source patch consistency loss). RankSim regularization is applied on intermediate features using segmentation-derived or source-derived targets to enforce rank consistency across patches.

## Results

Benchmarked on Swiss ALS LiDAR ground truth (10,000 samples, 64x64 patches):

| Method | SSIM | PSNR |
|---|---|---|
| Fast Guided Filter | 0.11 | 5.90 |
| Deep Anisotropic Diffusion (DADA) | 0.14 | 7.35 |
| Pix-Transform | 0.15 | 7.50 |
| **Pix-Transform + RankSim (ours)** | **0.19** | **7.55** |

RankSim-regularized PixTransform produces perceptually superior outputs (higher SSIM and PSNR) compared to all baselines.

## Data

| Source | Description | Resolution |
|---|---|---|
| [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) | Multispectral satellite imagery (RGB bands) | 10 m |
| Coarse VHM | GEDI spaceborne LiDAR + Sentinel-2 fusion | ~40 m effective |
| [Dynamic World](https://dynamicworld.app/) | Land cover segmentation (Google Earth Engine) | 10 m |
| Swiss ALS LiDAR | Airborne laser scanner ground truth (evaluation only) | 10 m |

The input data is stored in HDF5 format with keys: `guide` (Sentinel-2), `source` (coarse VHM), `gee`/`segmentation` (Dynamic World), `eval` (ALS target).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python pix_transform_ranksim.py \
    --data_dir path/to/data.h5 \
    --ranksim_weight 0.3 \
    --ranksim_target source \
    --lr 0.001 \
    --batchsize 32 \
    --scaling 4 \
    --index_s 0 \
    --index_f 10
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | — | Path to HDF5 data file |
| `--ranksim_weight` | `0.3` | Weight for RankSim regularization (0 = disabled) |
| `--ranksim_target` | `source` | RankSim target signal (`source`, `segmentation`, `rich_segmentation`, `binned_target`, `y_pred`) |
| `--positional_encoding` | `0` | Use sinusoidal positional encoding (1) or spatial coordinates (0) |
| `--lr` | `0.001` | Learning rate |
| `--batchsize` | `32` | Batch size |
| `--scaling` | `4` | Downsampling factor of coarse VHM |
| `--index_s` / `--index_f` | `0` / `1` | Start/end sample indices |
| `--filename` | `predictions` | Output filename prefix |

## Project Structure

```
├── pix_transform_ranksim.py        # PixTransform + RankSim implementation
├── preprocess.ipynb                # Sentinel-2 & ALS data preprocessing
├── Refinement_of_Large_scale_Vegetation_Height_Maps.pdf
├── requirements.txt
└── README.md
```

## Acknowledgements

This project builds on:

- [PixTransform](https://github.com/prs-eth/PixTransform) — self-supervised guided super-resolution via pixel-wise transforms
- [RankSim](https://github.com/BorealisAI/ranksim-imbalanced-regression) — ranking similarity regularizer for imbalanced regression
- [Multidimensional Positional Encoding](https://github.com/tatp22/multidim-positional-encoding) — 2D sinusoidal positional encoding
