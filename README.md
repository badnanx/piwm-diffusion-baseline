# PIWM Diffusion Baseline

A staged, extrinsic PIWM + latent diffusion baseline for Lunar Lander image generation.

Compares two autoencoder paradigms (continuous VAE vs VQ-VAE) under the same PIWM-aligned training setup, with P4 compositional generation (separate lander crop + background) evaluated on both. See [docs/paradigm_comparison.md](docs/paradigm_comparison.md) for full results.

See [docs/](docs/) for architecture details and results.

## Setup

```bash
cd piwm-diffusion-baseline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Data should be at `../data/lunar/extracted/lunar/{lunartrain,lunartest}` — each file is a `.npz` with keys `imgs (T,100,150,3)`, `states (T,8)`, `acts (T,)`.

## Running the Pipeline

All training is driven through `make`. The `PYTHON` variable must point to your venv:

```bash
# If .venv is active, just use:
make full-fast-crop RUN_DIR=outputs/my_run_v1 EPOCHS=20 TRAIN_FILES=40 TEST_FILES=10 AE_BATCH=16

# Or pass the path explicitly:
make full-fast-crop PYTHON=.venv/bin/python RUN_DIR=outputs/my_run_v1 ...
```

### Paradigm A — Continuous VAE (recommended starting point)

```bash
make full-fast-crop \
  RUN_DIR=outputs/paradigm_a_v1 \
  EPOCHS=20 TRAIN_FILES=40 TEST_FILES=10 AE_BATCH=16
```

### Paradigm B — VQ-VAE

```bash
make full-fast-crop-vq \
  RUN_DIR=outputs/paradigm_b_v1 \
  EPOCHS=20 TRAIN_FILES=40 TEST_FILES=10 AE_BATCH=16
```

The pipeline prints a stage header as each step starts:

```
=== STAGE 1/9: AE (continuous VAE + crop-24 loss) ===
=== STAGE 2/9: Physical Encoder (z -> f) ===
...
=== STAGE 9/9: Rollout Eval (real vs generated, crop-MSE) ===
```

### P4 Compositor (on top of an existing run)

```bash
make crop-ae export-crop-latents crop-ddpm p4-eval \
  RUN_DIR=outputs/paradigm_a_v1 CROP_EPOCHS=20 DEVICE=cuda
```

This adds CropVAE + CropDDPM + compositor eval to any existing AE/physical/dynamics checkpoint.

### Re-running individual stages

```bash
make ae-crop24 RUN_DIR=outputs/my_run_v1 EPOCHS=30   # retrain AE only
make rollout   RUN_DIR=outputs/my_run_v1              # re-run rollout eval
```

## Key Outputs

After a full run, inspect:

| File | What it shows |
|---|---|
| `*/ae/recon_best_boxed.png` | AE reconstructions with 24px lander crop boxes |
| `*/physical_eval/scatter.png` | R² per physical dim (x, y, θ) |
| `*/dynamics_eval/overlay.png` | True (green) vs predicted (red) lander position |
| `*/random_samples/samples.png` | Conditional diffusion samples |
| `*/rollout/rollout_real_vs_generated.png` | Real future frame vs PIWM+DDPM generated |
| `*/rollout/summary.json` | Crop-MSE, image-MSE, and constraint checker metrics |
| `*/crop_ae/recon_best.png` | CropVAE lander patch reconstructions |
| `*/p4_eval/p4_components.png` | 4-panel: background / generated crop / composite / real |
| `*/p4_eval/p4_rollout_real_vs_generated.png` | Real vs P4 composite full frame |
| `*/p4_eval/metrics.json` | P4 crop-MSE and constraint checker (centroid error, detection rate) |

Key metrics: `generated_crop_mse` in `rollout/summary.json` for baseline quality; `centroid_err_vs_true_px` in `p4_eval/metrics.json` and `rollout/summary.json` for physical positioning accuracy.

## Makefile Variables

| Variable | Default | Description |
|---|---|---|
| `PYTHON` | `python` | Python executable (set to `.venv/bin/python` if not activated) |
| `RUN_DIR` | `outputs/laptop_fast_v1` | Where all outputs go |
| `EPOCHS` | `3` | Epochs per stage |
| `TRAIN_FILES` | `24` | Number of training trajectory files |
| `TEST_FILES` | `6` | Number of test trajectory files |
| `LATENT_DIM` | `48` | AE latent dimension |
| `AE_BATCH` | `24` | AE batch size |
| `STATE_INDICES` | `0 1 4` | Which state dims to use (x, y, θ) |
| `STATE_WEIGHT` | `1.0` | P1 partitioning strength |
| `CROP_WEIGHT` | `1.0` | Lander crop loss weight |
| `CROP_SIZE` | `24` | Crop size in pixels |
| `CROP_EPOCHS` | `20` | Epochs for CropVAE and CropDDPM (separate from `EPOCHS`) |

## Architecture Overview

See [docs/architecture.md](docs/architecture.md) for a full description.

```
image → AE encoder → z (48-dim)
                      ↓
              physical encoder E_p
                      ↓
              f = [x, y, θ]  (physical state)
                      ↓
              dynamics model
                      ↓
              f̂_{t+2}  (predicted future state)
                      ↓
              conditional DDPM (in latent space)
                      ↓
              ẑ → frozen AE decoder → generated image
```
