# Paradigm Comparison: A vs B

## Goal

Compare two autoencoder paradigms in the same PIWM + latent diffusion pipeline. Everything downstream (physical encoder, dynamics, DDPM) is identical — only Stage 1 differs.

The long-term plan:

| Run | Description |
|---|---|
| Paradigm A | Continuous VAE + extrinsic physical encoder |
| Paradigm B | VQ-VAE + extrinsic physical encoder |
| Paradigm A + P4 | A + compositional crop generation |
| Paradigm B + P4 | B + compositional crop generation |

## Paradigm A Results — `outputs/laptop_p1_xyt_v1`

**Setup:** 20 epochs, 40 train files, 10 test files, latent_dim=48, state_indices=[0,1,4] (x, y, θ), state_weight=1.0, crop_weight=1.0

### Physical Encoder R²

| Dim | R² | Notes |
|---|---|---|
| x | 0.650 | Working — dots follow diagonal |
| y | 0.605 | Working — dense cluster at y≈1.4 (spawn point) |
| θ | 0.157 | Weak — angle hard to infer from single frame |

R² for y is partially inflated: most frames have the lander near the top (y≈1.4, spawn position). The model partly learns "y is probably ~1.4" for a large chunk of the dataset.

### Dynamics R²

| Dim | R² | Notes |
|---|---|---|
| x | 0.639 | Slightly worse than physical encoder (predicting future is harder) |
| y | 0.580 | Same pattern |
| θ | 0.140 | Near-zero, same root cause as above |

### Rollout Metrics (full episode distribution, 928 triplets)

| Metric | Value |
|---|---|
| dynamics_gt_mse | 0.163 |
| generated_image_mse | 0.041 |
| deterministic_dp_image_mse | 0.034 |
| generated_crop_mse | 0.024 |
| deterministic_dp_crop_mse | 0.022 |

The deterministic decoder path (f → D_p → D_v) still outperforms DDPM on crop-MSE. This is expected — the DDPM generates a full 100×150 frame from 3 numbers, so background noise dominates. P4 (crop-only DDPM) is the fix.

### Learned Physics Parameters (dynamics model)

| Parameter | Learned value | Expected sign |
|---|---|---|
| main_power | 0.507 | positive ✓ |
| side_power | 0.736 | positive ✓ |
| angular_power | 0.786 | positive ✓ |
| gravity | -0.924 | negative ✓ |

Signs are all correct. Magnitudes are underfit (expected ~1.0 for main_power) but directionally right.

## Paradigm B Results — `outputs/laptop_vq_crop24_v1`

*Pending — run in progress.*

## Key Differences to Watch For

When B finishes, compare:

1. **AE reconstruction quality** — `*/ae/recon_best_boxed.png`. VQ-VAE often produces sharper reconstructions due to the discrete bottleneck.
2. **Codebook utilization** — check perplexity in training logs. Low perplexity = codebook collapse (bad).
3. **Physical encoder R²** — VQ-VAE has a harder bottleneck so physical info may be harder to recover.
4. **Rollout crop-MSE** — the bottom-line metric for lander visibility.
