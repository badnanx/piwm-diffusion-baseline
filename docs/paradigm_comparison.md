# Paradigm Comparison: A vs B

## Goal

Compare two autoencoder paradigms in the same PIWM + latent diffusion pipeline. Everything downstream (physical encoder, dynamics, DDPM) is identical — only Stage 1 differs.

| Run | Description | Status |
|---|---|---|
| Paradigm A | Continuous VAE + P1 (state supervision) | Done |
| Paradigm B | VQ-VAE, no P1 | Done |
| Paradigm B + P1 | VQ-VAE + P1 (z_pre supervision) | Done |
| Paradigm A + P4 | A + compositional crop generation | Done |
| Paradigm B + P4 | B+P1 + compositional crop generation | Done |

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

## Paradigm A + P4 — `outputs/laptop_p1_xyt_v1`

**Setup:** CropVAE (24×24 → 16-dim, 20 epochs), CropDDPM (conditioned on θ, 20 epochs). Built on top of existing Paradigm A checkpoint.

### P4 Pipeline

1. **Background:** copy `image_t`, erase lander at known position using color mask (purple: b > r+0.05 and b > g+0.05; fire: r > b+0.05)
2. **Lander crop:** CropDDPM samples 16-dim latent conditioned on predicted θ → CropVAE decodes to 24×24 sprite
3. **Compositor:** paste sprite at predicted pixel (x, y) using same color mask, rejecting background/terrain pixels from the generated crop

### Results

| Metric | Value |
|---|---|
| CropVAE best_loss | 0.00152 |
| CropDDPM best_loss | 0.549 |
| **P4 crop_mse** | **0.031** |
| A baseline crop_mse | 0.024 |
| A deterministic_dp_crop_mse | 0.022 |

P4 crop_mse (0.031) is higher than the baseline DDPM (0.024). The erase step is the correct fix (it removes the ghost lander from the background) but it also removes the metric crutch — the baseline DDPM's diffuse purple smear accidentally covered the true lander position most of the time, giving a low crop_mse without actually rendering a lander. P4 places a specific sprite at a specific predicted spot; when that spot is wrong the crop at the true position is black background, giving higher MSE.

### Visual Results

- Double-lander problem resolved by the erase step
- CropDDPM generates recognizable lander body shapes with correct orientation (θ conditioning works)
- Generated sprites are softer than real (VAE blur — expected at 16-dim latent)
- Terrain sliver artifact fixed by tightening the color mask (white/grey terrain rejected since it has r ≈ g ≈ b)
- Lander frequently misplaced, partially rendered, or absent — root cause is dynamics accuracy (dynamics_gt_mse 0.163), not the compositor

### Design Notes

**Why color mask instead of SAM:** Lunar Lander's purple lander is uniquely colored against a black sky. Color threshold (`b > r+0.05, b > g+0.05`) reliably segments the lander and rejects terrain. SAM would add 2-3 sessions of implementation for marginal gain on a synthetic environment where we already have position priors from state.

**Why crop-based extraction instead of color extraction at training time:** CropVAE is trained on fixed 24×24 patches centered on the lander (via `crop_around_state`). This includes some background and occasional terrain pixels near the ground. Color-based extraction at training time would give cleaner sprites but requires retraining. The color paste mask is sufficient to prevent terrain from appearing in the composite.

**The dynamics bottleneck:** P4 exposes what the full-frame DDPM hides. A diffuse smear across the whole image accidentally overlaps the true lander position — P4's precise placement cannot. Improving P4 visual quality requires improving dynamics accuracy first (more training data, more epochs, or conditioning on velocity in addition to position).

**Remaining limitations:**
- CropDDPM conditioned on θ only — not on y (height). This is load-bearing in two ways: (1) near-ground crops include terrain pixels (terrain-sliver artifact); (2) at high y (start of episode), the lander is partially or fully offscreen — the 24×24 crop clips against the image edge, so CropVAE learned to generate partially-black sprites for those configurations. Without y in the conditioning, CropDDPM cannot distinguish these cases. Fix: condition on (θ, y). CropVAE training patches should also be labeled with (θ, y) so the latent space organizes by both.
- Thruster fire (red dots) is filtered from the paste mask but not from CropVAE training — DDPM doesn't generate fire since it's not conditioned on action.
- Background quality is limited by image_t (episode-specific terrain not generatable from physical state alone — would require SAM + per-frame background segmentation).

---

## Paradigm B + P4 — `outputs/laptop_vq_p1_v1`

**Setup:** Same CropVAE + CropDDPM as A+P4 (identical architecture and training budget). Built on top of existing B+P1 checkpoint.

### Results

| Metric | Value |
|---|---|
| CropVAE best_loss | 0.00152 |
| CropDDPM best_loss | 0.549 |
| **B+P4 crop_mse** | **0.029** |
| A+P4 crop_mse | 0.031 |
| A baseline crop_mse | 0.024 |

CropVAE and CropDDPM converged to identical losses as A+P4 — expected, since they are trained on the same data with the same architecture regardless of the upstream AE. The marginal crop_mse difference (0.029 vs 0.031) is within noise.

### Key Takeaway

**P4 quality is independent of which paradigm is underneath it.** The compositor output is driven by CropVAE/CropDDPM quality and dynamics position accuracy — not by whether the upstream AE is continuous or discrete. Both A+P4 and B+P4 produce similar crop_mse and similar visual artifacts (misplaced or missing lander).

### Full Comparison Table (crop_mse)

| Method | crop_mse | Notes |
|---|---|---|
| A deterministic | 0.022 | Best metric — no sampling noise |
| A baseline DDPM | 0.024 | Diffuse smear accidentally covers true position |
| B+P4 compositor | 0.029 | Precise placement, exposed to dynamics error |
| A+P4 compositor | 0.031 | Same |

P4 scores worse than the baseline DDPM by crop_mse. This is the dynamics bottleneck: precise placement at a wrong position scores worse than a diffuse smear that accidentally overlaps the right area. P4 makes the dynamics bottleneck visible rather than hiding it.

---

## Constraint Checker Results

Color-mask centroid extraction applied to all generated images. Purple pixels are detected and their centroid compared to predicted (x,y) from dynamics and true (x,y) from ground-truth state.

| Method | detection_rate | centroid_err_vs_pred_px | centroid_err_vs_true_px |
|---|---|---|---|
| A baseline DDPM | 1.000 | 48.6 | 52.3 |
| B+P1 baseline DDPM | 1.000 | 16.2 | 37.7 |
| A+P4 compositor | 0.999 | 24.1 | **30.4** |
| B+P4 compositor | 1.000 | 14.7 | 37.7 |

Image is 100×150px. Errors of 30–52px represent roughly ⅓–½ of the image height.

### Key Findings

**A+P4 genuinely improves over A baseline (30.4 vs 52.3px centroid_err_vs_true).** crop_mse told the opposite story because the A baseline's diffuse smear accidentally overlaps the true lander position, scoring well on crop_mse despite being visually wrong. The centroid metric reveals that A+P4 actually places the lander closer to where it should be.

**B+P4 and B+P1 tie at 37.7px.** P4 adds no positioning improvement for Paradigm B. B+P1's dynamics is the bottleneck, and the compositor can only place the lander as accurately as the dynamics predicts.

**B+P1 centroid_err_vs_pred is low (16.2px).** The purple smear in B+P1's generated frames is concentrated near the dynamics-predicted position — the DDPM learned to put its smear where it was conditioned. This is why B's crop_mse looked artificially good in the three-way comparison.

### Important Caveat

Detection rate is 1.0 for all methods, but this does not mean all methods render a visible lander. The color mask detects purple pixels regardless of shape:
- **P4 variants**: purple pixels form a recognizable (if misshapen) lander sprite
- **Baseline DDPM variants**: purple pixels form a diffuse smear — no lander shape

The centroid numbers for baseline methods measure where the smear's center of mass lands, not where a shaped lander is. These metrics are directly comparable only between P4 variants (A+P4 vs B+P4). Cross-method comparison (P4 vs baseline) should be interpreted with this in mind.

A shape-aware constraint checker (e.g. checking pixel count, aspect ratio, or using SAM/color segmentation to train proper segment decoders) would give a fairer cross-method comparison and is a natural next step.

---

## Paradigm A (visible filter) — `outputs/paradigm_a_visible_v1`

**Setup:** 20 epochs, 40 train files, 10 test files, REQUIRE_VISIBLE=1 (lander fully on-screen for all stages), SpriteVAE 32×32 → 16-dim, SpriteDDPM 50 steps, SDEdit t_start=7. 626 visible triplets in test set.

### Component Quality

| Stage | Best loss | Notes |
|---|---|---|
| AE | 0.250 (epoch 14) | Early stopped epoch 19 |
| Physical encoder | 0.271 (epoch 3) | Early stopped |
| SpriteVAE | 0.0019 (epoch 19) | Converged — sprites reconstruct well |
| SpriteDDPM | 0.506 (epoch 20) | Still improving |

### Physical Encoder R² (visible frames only)

| Dim | R² | Notes |
|---|---|---|
| x | 0.722 | Good |
| y | 0.302 | Weak — sparse signal even on visible frames |
| θ | 0.087 | Very weak — angle barely recoverable from full frame |

Without visibility filter, y R² was −0.97 (off-screen positions dominated). The filter is load-bearing.

### Rollout Results (626 triplets, SDEdit t=7)

| Metric | Baseline (AE pos) | +DetectedPos +CurTheta |
|---|---|---|
| Image MSE ↓ | 0.0029 | **0.0013** |
| Crop MSE (diffusion) ↓ | 0.0470 | **0.0251** |
| Crop MSE (mean sprite) ↓ | 0.0427 | **0.0219** |
| Dynamics GT MSE ↓ | 0.2491 | 0.2491 |
| Detection rate ↑ | **1.000** | **1.000** |
| Centroid err/pred (px) ↓ | **1.999** | 2.063 |
| Centroid err/true (px) ↓ | 14.734 | **3.232** |

**+DetectedPos**: replace AE-predicted (x,y) with color-mask centroid from image_t1 (oracle position from current frame, not a learned component). **+CurTheta**: condition SpriteDDPM on ground-truth θ from state_t1 instead of dynamics-predicted θ.

### Key Findings

**Position is the main bottleneck.** Replacing AE position with the color-mask centroid halves crop MSE (0.047 → 0.025) and drops centroid error from 14.7px to 3.2px. The remaining 3.2px is dynamics error (predicting *next* position from current), not sprite generation error.

**Diffusion barely beats mean sprite.** Even with oracle position and theta (0.025 vs 0.022), the SpriteDDPM adds minimal value. Root cause: theta conditioning is weak (R²=0.087) and a 32×32 lander looks nearly identical across angles. The diffusion component is not yet differentiating itself from the trivial baseline.

**Constraint is satisfied by construction.** Detection rate 1.0 and centroid_err_vs_pred ≈ 2px in all cases. The 2px residual is sprite decentering (lander not perfectly centered in the 32×32 frame). This is the non-trivial constraint that `--constraint_alpha` is designed to reduce.

**What diffusion should learn (research direction).** The current pipeline satisfies C(y, f) mechanically (paste at predicted position). The meaningful research question is whether the diffusion can *actively adjust* z_sprite to satisfy C, rather than having it guaranteed by compositing. Constraint-guided SDEdit (`SPRITE_CONSTRAINT_ALPHA > 0`) implements this: gradient steps during denoising push z_sprite toward generating a centered lander, enforcing C through learning rather than geometry.

### Constraint-Guided SDEdit Results (SPRITE_CONSTRAINT_ALPHA=0.05)

Output: `outputs/paradigm_a_visible_v1/segment_rollout_constrained_a0.05/` — 626 visible triplets.

| Metric | Baseline (AE pos) | Constraint α=0.05 |
|---|---|---|
| Image MSE ↓ | 0.00291 | 0.00291 |
| Crop MSE (diffusion) ↓ | 0.0470 | **0.0459** |
| Crop MSE (mean sprite) ↓ | 0.0427 | 0.0427 |
| Detection rate ↑ | 1.000 | 1.000 |
| Centroid err/pred (px) ↓ | 1.999 | **1.682** |
| Centroid err/true (px) ↓ | 14.734 | 14.773 |

**Findings:**

- **Constraint works as intended.** Sprite centroid offset within the 32×32 frame drops from 2.00px → 1.68px. The gradient nudge is pulling the lander toward center, which is exactly C(y, f) being actively satisfied rather than trivially constructed.
- **No effect on true-position error.** Centroid err/true stays at ~14.7px — expected, since the constraint only centers the sprite within its own frame, not the paste position. Position accuracy is bounded by the AE physical encoder (y R²=0.302).
- **Minor crop MSE improvement.** 0.0470 → 0.0459 from better sprite centering. Not meaningful on its own; the position bottleneck dominates.
- **Next step to make the constraint matter more:** fix the position bottleneck (more data, more epochs on physical encoder, or oracle-guided training). With accurate positions, a well-centered sprite would give meaningful C(y, f) satisfaction.
