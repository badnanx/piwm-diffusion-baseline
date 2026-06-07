# Architecture

## Pipeline Overview

The full pipeline has 9 stages, run end-to-end via `make full-fast-crop`:

```
Stage 1: Visual AE          — learns to compress 100×150 images into z ∈ R^48
Stage 2: Physical Encoder   — learns E_p(z) → f = [x, y, θ]
Stage 3: Eval Physical      — measures R² per physical dim
Stage 4: Dynamics Model     — learns f_t, f_{t+1}, a → f̂_{t+2}
Stage 5: Eval Dynamics      — measures R² for future state prediction
Stage 6: Export Latents     — encodes full train/test sets to z, f pairs
Stage 7: DDPM               — trains conditional denoiser: z | f
Stage 8: Samples            — generates images from random test conditions
Stage 9: Rollout Eval       — full loop: image → z → f → dynamics → DDPM → image
```

## Extrinsic vs Intrinsic Autoencoding

This repo uses the **extrinsic** approach from the PIWM paper:

- **Extrinsic**: the AE is a visual autoencoder. A separate MLP (physical encoder) maps `z → f` after training. The AE is not directly supervised on physical state — only indirectly via the P1 state loss on `mu[:, :k]`.
- **Intrinsic**: a single encoder maps the image directly to a combined `[physical | visual]` latent. Not used here.

The extrinsic approach is cleaner for comparison — both paradigms (A and B) use the same physical encoder and dynamics code, differing only in the AE.

## PIWM Principle Alignment

### P1 — Functionally Organized Latent Space

Implemented via `state_weight > 0` in the AE loss:

```
state_loss = MSE(mu[:, 0], x) + MSE(mu[:, 1], y) + MSE(mu[:, 2], θ)
```

The first 3 dims of the latent are directly supervised to equal the physical state. The KL loss is applied only to the remaining visual dims (`mu[:, 3:]`) so it doesn't fight the state constraint.

**Status: active** (`STATE_WEIGHT=1.0` by default)

### P2 — Invariant / Equivariant Representations

Would require the latent to be invariant to irrelevant transformations (lighting, background variation) and equivariant to position/rotation. Not yet implemented — Lunar Lander has a fixed background so this matters less here.

**Status: not implemented**

### P3 — Multi-Level Supervision

Partially implemented:
- **Crop loss**: 24px MSE around the lander region during AE training — weak spatial supervision
- **State supervision**: physical encoder and dynamics both trained against ground truth state

Planned: temporal smoothness loss `L_smooth = Σ||f_t - 2f_{t+1} + f_{t+2}||²`, and noisy-state ablation using `noisy_states_2/5/10` dataset keys.

**Status: partial**

### P4 — Compositional Output Generation

The DDPM currently generates full 100×150 frames from 3 condition numbers. This makes lander placement imprecise because the decoder has to place the lander based on a noisy latent rather than explicit coordinates.

Planned P4 approach:
1. Train a separate 24×24 crop AE on lander patches
2. Train a DDPM conditioned on θ to generate crop latents
3. Use (x, y) from the dynamics model to place the crop in pixel space
4. Composite onto a copied background frame

**Status: planned**

## Loss Functions

### AE (`piwm_vae_loss`)
```
L = MSE(recon, image)                          # full-frame reconstruction
  + kl_weight  * KL(mu[:,3:], logvar[:,3:])    # KL on visual dims only
  + state_weight * MSE(mu[:,:3], [x, y, θ])   # P1 partitioning
  + crop_weight * MSE(crop(recon), crop(image))# 24px lander crop
```

### Physical Encoder
```
L = MSE(E_p(z), f_true) + 0.1 * MSE(D_p(E_p(z)), z)
```

### Dynamics
```
L = MSE(f̂_{t+2}, f_encoded_{t+2}) + MSE(f̂_{t+2}, f_true_{t+2})
```

### DDPM (denoising score matching)
```
L = MSE(ε_θ(z_noisy, t, f), ε)
```

## Two Paradigms

| | Paradigm A | Paradigm B |
|---|---|---|
| AE type | Continuous VAE | VQ-VAE |
| Latent | Gaussian (μ, σ), reparameterized | Discrete codebook (512 codes) |
| Regularization | KL divergence | Commitment loss |
| P1 partitioning | Yes (state_weight on μ) | Not directly applicable |
| Downstream stages | Physical encoder, dynamics, DDPM | Same |
| Make target | `full-fast-crop` | `full-fast-crop-vq` |

See [paradigm_comparison.md](paradigm_comparison.md) for results.
