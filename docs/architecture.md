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

Implemented. A separate CropVAE + CropDDPM generates the lander sprite; the compositor pastes it at the dynamics-predicted position onto a background copied from `image_t`.

**Pipeline:**
1. **Background:** copy `image_t`, erase lander using color mask (purple: `b > r+0.05` and `b > g+0.05`; fire: `r > b+0.05`)
2. **Lander sprite:** CropDDPM samples a 16-dim latent conditioned on predicted θ → CropVAE decodes to 24×24 patch
3. **Compositor:** paste sprite at predicted pixel (x, y), rejecting non-lander pixels from the generated crop via color mask

**Known limitations:**
- CropDDPM conditioned on θ only, not y. This affects the output in two ways: (1) near-ground crops include terrain pixels; (2) at high y (start of episode), the lander is partially or fully offscreen — the 24×24 crop clips the image edge, so CropVAE learned partially-black sprites for those states. Fix: condition on (θ, y).
- Background is copied from `image_t` — episode-specific terrain cannot be generated from physical state alone.
- Positioning accuracy is bounded by dynamics accuracy (30–52px error on a 100×150px image).

**Make targets:** `crop-ae export-crop-latents crop-ddpm p4-eval`

**Status: implemented** (evaluated on both Paradigm A and B+P1 — see [paradigm_comparison.md](paradigm_comparison.md))

### SegmentVAE Pipeline (Paradigm A + visible filter)

A revised P4-inspired compositor that keeps a strict visibility contract throughout training and eval and adds constraint-guided generation.

**Visibility filter** — frames where the lander has fewer than 30 purple pixels or whose bounding box touches the image edge are excluded from all stages: AE, physical encoder, dynamics, SpriteVAE, SpriteDDPM, rollout. This prevents off-screen positions from corrupting the physical encoder (which caused y R²=−0.97 without the filter).

**SpriteVAE** — encodes/decodes 32×32 lander sprites into 16-dim z_sprite. Trained on sprites extracted from visible frames via color mask. Separate from the full-frame AE.

**SpriteDDPM** — denoising diffusion model over z_sprite, conditioned on predicted θ (normalized). 50 denoising steps.

**SDEdit** — for temporal continuity, encode the current frame's sprite → z0, forward-diffuse to t_start=7, then reverse-denoise conditioned on θ_pred. This seeds generation from the current lander shape rather than pure noise.

**Compositing** — clean background from image_t1 (zero all purple pixels), decode z_sprite → 32×32 sprite, paste centered at pixel(x_pred, y_pred) from dynamics.

**Constraint checker C(y, f)**

Physical consistency: detected position in the generated image should match the predicted physical state. Implemented via hard color mask:

```
centroid(y) = mean pixel of { p : b > r+0.051, b > g+0.051, b > 0.10 }
C(y, f)     = || centroid(y) - pixel_coords(f_pred_x, f_pred_y) ||
```

In the standard pipeline C is trivially ~0 because the sprite is pasted at pixel_coords(f_pred). The non-trivial constraint is that the decoded sprite's purple pixels must be *centered within the 32×32 frame* (otherwise the composite centroid drifts from the paste point).

**Constraint-guided SDEdit**

With `SPRITE_CONSTRAINT_ALPHA > 0`, after each denoising step a gradient step minimizes:

```
C_center(z_sprite) = || centroid(soft_purple(decode(z_sprite))) - (S/2, S/2) ||²
```

where `soft_purple` is a differentiable relaxation of the hard threshold (sigmoid with T=0.02). This directly implements the research direction: diffusion adjusts latents to satisfy the physical constraint rather than having it satisfied by construction.

```
dz_norm ← dz_norm - α · ∂C_center/∂z_norm
```

**Make targets:** `full-segment segment-rollout` with `SPRITE_CONSTRAINT_ALPHA=0.05`

**Status: implemented** (`outputs/paradigm_a_visible_v1` — see [paradigm_comparison.md](paradigm_comparison.md))

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
