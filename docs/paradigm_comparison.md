# Paradigm Comparison: A vs B

## Goal

Compare two autoencoder paradigms in the same PIWM + latent diffusion pipeline. Everything downstream (physical encoder, dynamics, DDPM) is identical — only Stage 1 differs.

| Run | Description | Status |
|---|---|---|
| Paradigm A | Continuous VAE + P1 (state supervision) | Done |
| Paradigm B | VQ-VAE, no P1 | Done |
| Paradigm B + P1 | VQ-VAE + P1 (z_pre supervision) | Done |
| Paradigm A + P4 | A + compositional crop generation | Pending |
| Paradigm B + P4 | B+P1 + compositional crop generation | Pending |

---

## Three-Way Rollout Comparison (928 triplets each)

| Metric | A (VAE+P1) | B (VQ, no P1) | B+P1 (VQ+P1) |
|---|---|---|---|
| **dynamics_gt_mse** | **0.163** | 0.260 | 0.235 |
| generated_image_mse | 0.041 | 0.031 | **0.028** |
| generated_physical_vs_pred_mse | 0.114 | **0.021** | 0.049 |
| generated_physical_vs_gt_mse | **0.239** | 0.266 | 0.278 |
| deterministic_dp_image_mse | 0.034 | 0.026 | **0.026** |
| generated_crop_mse | 0.024 | **0.007** | 0.008 |
| deterministic_dp_crop_mse | 0.022 | **0.009** | 0.010 |

### Key Takeaways

- **A wins on dynamics accuracy** (`dynamics_gt_mse` 0.163 vs 0.235/0.260). Continuous latents are easier to learn physical dynamics on than discrete codes.
- **B's low crop_mse is misleading.** The physical encoder in B (no P1) had collapsed R² ≈ 0 (x=0.042, y=-0.073). When the physical encoder outputs near-zero for everything, the dynamics and DDPM also converge to near-zero — they appear "consistent" but are consistently wrong. The crop_mse looks good because both generated and real crops happen to be mostly dark background.
- **P1 genuinely helped B.** `dynamics_gt_mse` improved from 0.260 → 0.235, and `generated_physical_vs_pred_mse` increased from 0.021 → 0.049 — meaning the generated latent is now actually trying to encode position rather than collapsing to zero.
- **VQ-VAE + P1 structural limitation.** P1 supervises `z_pre` (before codebook lookup), but the full pipeline uses `z_quant` (after). VQ quantization can discard the position structure even if `z_pre` was perfectly aligned. The discrete bottleneck is the likely root cause of A's dynamics advantage.
- **Neither paradigm visually renders the lander.** Both produce purple smears/smoke. Root cause: the DDPM generates a full 48-dim frame latent from only 3 numbers (x, y, θ). It averages out to a blur around where the lander probably is. **P4 (crop compositor) is the fix.**

---

## Paradigm A — `outputs/laptop_p1_xyt_v1`

**Setup:** 20 epochs, 40 train files, 10 test files, latent_dim=48, state_indices=[0,1,4] (x, y, θ), state_weight=1.0, crop_weight=1.0

### Physical Encoder R²

| Dim | R² | Notes |
|---|---|---|
| x | 0.650 | Working |
| y | 0.605 | Working — dense cluster at y≈1.4 (spawn point inflates score) |
| θ | 0.157 | Weak — angle is hard to infer from a single frame |

### Dynamics R²

| Dim | R² | Notes |
|---|---|---|
| x | 0.639 | Slightly worse than physical encoder (future harder than present) |
| y | 0.580 | Same pattern |
| θ | 0.140 | Near-zero, same root cause |

### Learned Physics Parameters

| Parameter | Learned | Expected sign |
|---|---|---|
| main_power | 0.507 | positive ✓ |
| side_power | 0.736 | positive ✓ |
| angular_power | 0.786 | positive ✓ |
| gravity | -0.924 | negative ✓ |

Signs correct. Magnitudes underfit (main_power expected ~1.0) but directionally right.

### Visual Result

Purple smear where lander should be. Background reconstruction is clean. DDPM generating full frame from 3 numbers averages out the lander position across all likely poses.

---

## Paradigm B (no P1) — `outputs/laptop_vq_crop24_v1`

**Setup:** Same as A but VQ-VAE, no state_weight (P1 disabled).

### Physical Encoder R² (collapsed)

| Dim | R² | Notes |
|---|---|---|
| x | ~0.042 | Collapsed — encoder learned nothing about position |
| y | ~-0.073 | Collapsed |
| θ | ~0.0 | Collapsed |

Root cause: VQ codebook has no spatial organization without P1. The discrete codes don't consistently map to spatial positions, so the physical encoder can't recover them.

### Learned Physics Parameters

| Parameter | Learned | Expected sign |
|---|---|---|
| main_power | 0.944 | positive ✓ |
| side_power | 0.875 | positive ✓ |
| angular_power | 0.783 | positive ✓ |
| gravity | 0.133 | **negative ✗** |

Gravity learned with wrong sign — dynamics model is compensating for meaningless physical features by overfitting to the wrong signal.

---

## Paradigm B+P1 — `outputs/laptop_vq_p1_v1`

**Setup:** Same as B but state_weight=1.0 (P1 on z_pre).

### AE Training

Best loss: 0.238 at epoch 4 (early stopped). VQ-VAE converges faster but to a worse reconstruction than continuous VAE.

### Physical Encoder

Best loss: 0.226 at epoch 5.

### Dynamics

Best loss: 0.242 at epoch 20. Did not converge — still improving at the end of training budget. More epochs would help.

### Learned Physics Parameters

| Parameter | Learned | Expected sign |
|---|---|---|
| main_power | 0.937 | positive ✓ |
| side_power | 0.875 | positive ✓ |
| angular_power | 0.783 | positive ✓ |
| gravity | 0.133 | **negative ✗** |

Gravity sign still wrong. Both B variants share this failure — the discrete latent makes it harder for dynamics to recover correct physics.

### Visual Result

Purple smoke/streak. Same failure mode as A. P1 did not fix the rendering — it only improved dynamics accuracy. The DDPM full-frame generation bottleneck is unchanged.

---

## Next Steps: P4 Compositional Generation

P4 replaces full-frame DDPM generation with a two-part compositor:

1. **Background:** copy `image_t` directly (no generation needed)
2. **Lander crop:** train a tiny CropVAE (24×24 → 16-dim) and a crop-only DDPM conditioned on θ
3. **Compositor:** decode crop latent → paste at predicted pixel (x, y) using a lander mask (threshold max RGB > 0.08)

All scripts are implemented (`train_crop_ae.py`, `export_crop_latents.py`, `train_crop_ddpm.py`, `eval_p4_compositor.py`). Run on top of existing A checkpoint:

```bash
make PYTHON=.venv/bin/python RUN_DIR=outputs/laptop_p1_xyt_v1 \
  crop-ae export-crop-latents crop-ddpm p4-eval
```

P4 is expected to fix the lander rendering problem since it sidesteps the "generate full frame from 3 numbers" bottleneck.
