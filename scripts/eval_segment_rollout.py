#!/usr/bin/env python
"""SegmentVAE rollout evaluation.

Pipeline per step:
  1. image_t1 → [Full AE] → z → [Phys Enc] → f_t  (use t1 as current frame)
  2. image_t  → [Full AE] → z → [Phys Enc] → f_t_prev
  3. dynamics(f_t_prev, f_t, action) → f_pred = [x, y, theta]
  4. background = clean_background(image_t1)  — zero all purple pixels
  5. sprite_z ~ SpritesDDPM(cond=theta_pred)
  6. sprite   = SpriteVAE.decode(sprite_z)
  7. (x_pred, y_pred) → pixel coords  via state_xy_to_pixel
  8. composite = paste_sprite(background, sprite, cx, cy)
  Metrics: crop MSE at true t+2 position, centroid error vs pred/true.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from piwm_diffusion.crop import (
    crop_around_state, state_xy_to_pixel,
    WORLD_H, HALF_WORLD_H, Y_OFFSET,
)
from piwm_diffusion.crop_ae import SpriteVAE
from piwm_diffusion.data import LunarTripletDataset
from piwm_diffusion.diffusion import ConditionalDenoiserMLP, DiffusionSchedule, sample_latents
from piwm_diffusion.dynamics import LunarSecondOrderDynamics
from piwm_diffusion.physical import PhysicalAutoencoder
from piwm_diffusion.sprite import clean_background, extract_sprite, paste_sprite
from piwm_diffusion.train_utils import device_from_arg, load_autoencoder, set_seed, write_json


def load_sprite_vae(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SpriteVAE(latent_dim=ckpt["latent_dim"], sprite_size=ckpt.get("sprite_size", 32)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def load_sprite_ddpm(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = ConditionalDenoiserMLP(
        latent_dim=ckpt["latent_dim"], cond_dim=ckpt["cond_dim"],
        hidden_dim=a["hidden_dim"], num_layers=a["num_layers"],
        time_dim=a.get("time_dim", 64),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    schedule = DiffusionSchedule(
        num_steps=a["diffusion_steps"], beta_start=a["beta_start"], beta_end=a["beta_end"],
        device=device,
    )
    return model, schedule, ckpt


def make_constraint_fn(sprite_vae, latent_mean, latent_std, alpha, sprite_size, n_steps=1):
    """Return a constraint_fn for use in sample_latents.

    After each SDEdit denoising step, takes n_steps gradient steps on z_norm to minimize
    the offset of the decoded sprite's purple centroid from the sprite center:
        C = ||centroid(soft_purple(decode(z_sprite))) - (S/2, S/2)||²

    This is the non-trivial version of C(y, f): the sprite's purple pixels must be
    centered so that the composite centroid lands at the predicted paste position.
    Differentiable through the SpriteVAE decoder via a soft sigmoid color mask.
    """
    center = (sprite_size - 1) / 2.0

    def constraint_fn(z_norm):
        for _ in range(n_steps):
            z_req = z_norm.clone().requires_grad_(True)
            with torch.enable_grad():
                z = z_req * latent_std + latent_mean
                sprite = sprite_vae.decode(z)                        # [B, 3, S, S]
                r, g, b = sprite[:, 0:1], sprite[:, 1:2], sprite[:, 2:3]
                T = 0.02                                              # sigmoid temperature
                mask = (torch.sigmoid((b - r - 0.051) / T) *
                        torch.sigmoid((b - g - 0.051) / T) *
                        torch.sigmoid((b - 0.10) / T))               # [B, 1, S, S]
                S = sprite_size
                xs = torch.linspace(0, S - 1, S, device=sprite.device).view(1, 1, 1, S).expand_as(mask)
                ys = torch.linspace(0, S - 1, S, device=sprite.device).view(1, 1, S, 1).expand_as(mask)
                w = mask / (mask.sum(dim=(-2, -1), keepdim=True) + 1e-8)
                cx = (w * xs).sum(dim=(-2, -1)).squeeze(1)           # [B]
                cy = (w * ys).sum(dim=(-2, -1)).squeeze(1)
                C = ((cx - center) ** 2 + (cy - center) ** 2).sum()
            grad = torch.autograd.grad(C, z_req)[0]
            z_norm = (z_norm - alpha * grad).detach()
        return z_norm

    return constraint_fn


def extract_centroid(img_chw, min_pixels=5):
    """Return (cx, cy) of purple pixels or (None, None)."""
    from piwm_diffusion.sprite import purple_mask
    mask = purple_mask(img_chw)
    if int(mask.sum()) < min_pixels:
        return None, None
    ys, xs = torch.where(mask)
    return xs.float().mean().item(), ys.float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--physical_checkpoint", required=True)
    parser.add_argument("--dynamics_checkpoint", required=True)
    parser.add_argument("--sprite_ae_checkpoint", required=True)
    parser.add_argument("--sprite_ddpm_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/segment_rollout")
    parser.add_argument("--state_indices", type=int, nargs="+", default=[0, 1, 4])
    parser.add_argument("--crop_size", type=int, default=24,
                        help="Crop size for crop-MSE metric (not the sprite size)")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_triplets_per_file", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_viz", type=int, default=16)
    parser.add_argument("--require_visible", action="store_true")
    parser.add_argument("--sdedit_t_start", type=int, default=-1,
                        help="SDEdit start step for sprites: encode the current sprite from "
                             "image_t1, forward-diffuse to this step, then denoise toward "
                             "theta_pred. -1 = pure noise (default).")
    parser.add_argument("--use_detected_position", action="store_true",
                        help="Use purple-pixel centroid of image_t1 for compositing position "
                             "instead of dynamics-predicted x,y. Diagnostic: isolates sprite "
                             "generation quality from position prediction quality.")
    parser.add_argument("--use_current_theta", action="store_true",
                        help="Condition DDPM on ground-truth theta from state_t1 instead of "
                             "dynamics-predicted theta. Diagnostic: isolates theta-conditioning "
                             "quality from dynamics quality.")
    parser.add_argument("--constraint_alpha", type=float, default=0.0,
                        help="Gradient step size for constraint-guided SDEdit. At each denoising "
                             "step, nudges z_sprite toward a centered lander: "
                             "C = ||centroid(soft_purple(decode(z))) - (S/2,S/2)||². "
                             "0 = disabled (default).")
    parser.add_argument("--constraint_steps", type=int, default=1,
                        help="Gradient steps per denoising step when constraint_alpha > 0.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = device_from_arg(args.device)
    max_triplets = None if args.max_triplets_per_file == 0 else args.max_triplets_per_file

    # --- Load models ---
    autoencoder, ae_ckpt = load_autoencoder(args.autoencoder_checkpoint, device)

    phys_ckpt = torch.load(args.physical_checkpoint, map_location=device, weights_only=False)
    phys_args = phys_ckpt["args"]
    state_indices = list(phys_ckpt["state_indices"])
    physical_model = PhysicalAutoencoder(
        latent_dim=int(phys_ckpt["latent_dim"]), state_dim=int(phys_ckpt["state_dim"]),
        hidden_dim=int(phys_args["hidden_dim"]), num_layers=int(phys_args["num_layers"]),
    ).to(device)
    physical_model.load_state_dict(phys_ckpt["model_state_dict"])
    physical_model.eval()

    dyn_ckpt = torch.load(args.dynamics_checkpoint, map_location=device, weights_only=False)
    dynamics = LunarSecondOrderDynamics(state_dim=int(dyn_ckpt["state_dim"])).to(device)
    dynamics.load_state_dict(dyn_ckpt["model_state_dict"])
    dynamics.eval()

    sprite_vae, vae_ckpt = load_sprite_vae(args.sprite_ae_checkpoint, device)
    sprite_size = int(vae_ckpt.get("sprite_size", 32))
    sprite_ddpm, sprite_schedule, ddpm_ckpt = load_sprite_ddpm(args.sprite_ddpm_checkpoint, device)

    cond_mean    = torch.from_numpy(ddpm_ckpt["cond_mean"]).to(device)
    cond_std     = torch.from_numpy(ddpm_ckpt["cond_std"]).to(device)
    latent_mean  = torch.from_numpy(ddpm_ckpt["latent_mean"]).to(device)
    latent_std   = torch.from_numpy(ddpm_ckpt["latent_std"]).to(device)

    constraint_fn = None
    if args.constraint_alpha > 0:
        constraint_fn = make_constraint_fn(
            sprite_vae, latent_mean, latent_std,
            args.constraint_alpha, sprite_size, args.constraint_steps,
        )

    # theta index within f vector (f = [x, y, theta] by default)
    xi = state_indices.index(0) if 0 in state_indices else None
    yi = state_indices.index(1) if 1 in state_indices else None
    ti = state_indices.index(4) if 4 in state_indices else None
    has_xy = xi is not None and yi is not None

    ds = LunarTripletDataset(
        args.data_dir, max_files=args.max_files,
        max_triplets_per_file=max_triplets,
        require_visible=args.require_visible, file_seed=args.seed,
    )
    loader     = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    viz_loader = DataLoader(ds, batch_size=args.num_viz, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(args.seed + 99))

    metrics = {
        "dynamics_gt_mse": [], "composite_image_mse": [],
        "composite_crop_mse": [], "deterministic_crop_mse": [],
    }
    cc_detected, cc_err_vs_pred, cc_err_vs_true = [], [], []
    all_real, all_composite, all_background = [], [], []

    @torch.no_grad()
    def run_batch(batch):
        image_t  = batch["image_t"].to(device)
        image_t1 = batch["image_t1"].to(device)
        image_t2 = batch["image_t2"].to(device)
        gt_f2    = batch["state_t2"].to(device)[:, state_indices]
        action   = batch["action_t1"].to(device)
        B, _, H, W = image_t.shape

        # Physical state from two consecutive frames
        z_t,  _ = autoencoder.encode(image_t)
        z_t1, _ = autoencoder.encode(image_t1)
        f_t,  _ = physical_model(z_t)
        f_t1, _ = physical_model(z_t1)
        f_pred   = dynamics(f_t, f_t1, action)

        # Predicted pixel position from dynamics
        fake_state = torch.zeros(B, 8, device=device)
        if has_xy:
            fake_state[:, 0] = f_pred[:, xi]
            fake_state[:, 1] = f_pred[:, yi]
        px_pred, py_pred = state_xy_to_pixel(fake_state, H, W)

        # Fix 1: override with detected purple centroid from image_t1
        # Uses the mask directly — no learning, no AE. Diagnostic for position quality.
        if args.use_detected_position:
            px_paste_list, py_paste_list = [], []
            for i in range(B):
                cx, cy = extract_centroid(image_t1[i])
                px_paste_list.append(float(cx) if cx is not None else float(px_pred[i]))
                py_paste_list.append(float(cy) if cy is not None else float(py_pred[i]))
            px_paste = torch.tensor(px_paste_list, device=device)
            py_paste = torch.tensor(py_paste_list, device=device)
            # Back-convert detected pixels to state-space for crop_around_state metric
            x_det = px_paste / W * 2.0 - 1.0
            world_y_det = (H - py_paste) / H * WORLD_H
            y_det = (world_y_det - Y_OFFSET) / HALF_WORLD_H
            f_xy_crop = torch.stack([x_det, y_det], dim=1)
        else:
            px_paste, py_paste = px_pred, py_pred
            f_xy_crop = None  # will use f_pred x,y

        # Fix 3: condition DDPM on ground-truth theta from current frame (state_t1)
        # instead of dynamics-predicted theta. Diagnostic for theta-conditioning quality.
        if args.use_current_theta and ti is not None:
            theta_src = batch["state_t1"].to(device)[:, 4:5]
        elif ti is not None:
            theta_src = f_pred[:, ti:ti+1]
        else:
            theta_src = torch.zeros(B, 1, device=device)
        theta_norm = (theta_src - cond_mean) / cond_std

        # SDEdit: seed from current sprite if enabled, otherwise pure noise.
        # Encodes the sprite visible in image_t1 and forward-diffuses to t_start,
        # then denoises conditioned on theta_pred — keeps continuity with current frame.
        x0_init = None
        sdedit_t = args.sdedit_t_start if args.sdedit_t_start > 0 else None
        if sdedit_t is not None:
            current_sprites = torch.stack([
                extract_sprite(image_t1[i], size=sprite_size)[0] for i in range(B)
            ])
            z0, _ = sprite_vae.encode(current_sprites)
            x0_init = (z0 - latent_mean) / latent_std

        sprite_z = sample_latents(
            sprite_ddpm, sprite_schedule, theta_norm, sprite_vae.latent_dim,
            x0_init=x0_init, t_start=sdedit_t,
            constraint_fn=constraint_fn,
        )
        sprites  = sprite_vae.decode(sprite_z)   # (B, 3, sprite_size, sprite_size)

        # Composite: clean background from image_t1, paste sprite at predicted pos
        composites  = []
        backgrounds = []
        for i in range(B):
            bg   = clean_background(image_t1[i])
            comp = paste_sprite(bg, sprites[i], float(px_paste[i]), float(py_paste[i]), sprite_size)
            backgrounds.append(bg)
            composites.append(comp)
        backgrounds = torch.stack(backgrounds)
        composites  = torch.stack(composites)

        return {
            "image_t2": image_t2, "gt_f2": gt_f2, "f_pred": f_pred,
            "composites": composites, "backgrounds": backgrounds,
            "px_pred": px_paste, "py_pred": py_paste,
            "f_xy_crop": f_xy_crop,
        }

    # Metrics pass
    with torch.no_grad():
        for batch in tqdm(loader, desc="rollout"):
            out = run_batch(batch)
            B, _, H, W = out["image_t2"].shape
            image_t2   = out["image_t2"]
            composites = out["composites"]
            gt_f2      = out["gt_f2"]
            f_pred     = out["f_pred"]

            metrics["dynamics_gt_mse"].append(F.mse_loss(f_pred, gt_f2).item())
            metrics["composite_image_mse"].append(F.mse_loss(composites, image_t2).item())

            # Deterministic baseline: zero latent = mean sprite in normalized space.
            # Crop MSE computed the same way as composite_crop_mse so the comparison is fair.
            if has_xy:
                mean_z = torch.zeros(B, sprite_vae.latent_dim, device=device)
                mean_sprites_b = sprite_vae.decode(mean_z)
                mean_comps = torch.stack([
                    paste_sprite(out["backgrounds"][i], mean_sprites_b[i],
                                 float(out["px_pred"][i]), float(out["py_pred"][i]), sprite_size)
                    for i in range(B)
                ])
                f_xy_det = out["f_xy_crop"] if out["f_xy_crop"] is not None \
                    else torch.stack([f_pred[:, xi], f_pred[:, yi]], dim=1)
                metrics["deterministic_crop_mse"].append(
                    F.mse_loss(crop_around_state(mean_comps, f_xy_det, args.crop_size),
                               crop_around_state(image_t2,   f_xy_det, args.crop_size)).item()
                )

            if has_xy:
                # Use detected/overridden position for crop if available, else dynamics
                f_xy = out["f_xy_crop"] if out["f_xy_crop"] is not None \
                    else torch.stack([f_pred[:, xi], f_pred[:, yi]], dim=1)
                metrics["composite_crop_mse"].append(
                    F.mse_loss(crop_around_state(composites, f_xy, args.crop_size),
                               crop_around_state(image_t2,   f_xy, args.crop_size)).item()
                )

                # True pixel positions
                true_state = torch.zeros(B, 8, device=device)
                true_state[:, 0] = gt_f2[:, xi]
                true_state[:, 1] = gt_f2[:, yi]
                px_true, py_true = state_xy_to_pixel(true_state, H, W)

                for i in range(B):
                    cx, cy = extract_centroid(composites[i])
                    cc_detected.append(cx is not None)
                    if cx is not None:
                        cc_err_vs_pred.append(np.sqrt(
                            (cx - float(out["px_pred"][i]))**2 +
                            (cy - float(out["py_pred"][i]))**2
                        ))
                        cc_err_vs_true.append(np.sqrt(
                            (cx - float(px_true[i]))**2 +
                            (cy - float(py_true[i]))**2
                        ))

            all_real.append(image_t2.cpu())
            all_composite.append(composites.cpu())
            all_background.append(out["backgrounds"].cpu())

    # Summary
    summary = {k: float(np.mean(v)) if v else None for k, v in metrics.items()}
    if cc_detected:
        summary["constraint_detection_rate"]        = float(np.mean(cc_detected))
        summary["constraint_centroid_err_vs_pred_px"] = float(np.mean(cc_err_vs_pred)) if cc_err_vs_pred else float("nan")
        summary["constraint_centroid_err_vs_true_px"] = float(np.mean(cc_err_vs_true)) if cc_err_vs_true else float("nan")
        summary["constraint_n_detected"] = int(sum(cc_detected))
    summary["num_triplets"]    = len(ds)
    summary["state_indices"]   = state_indices
    summary["sprite_size"]     = sprite_size
    summary["require_visible"]       = args.require_visible
    summary["sdedit_t_start"]        = args.sdedit_t_start
    summary["use_detected_position"] = args.use_detected_position
    summary["use_current_theta"]     = args.use_current_theta
    summary["constraint_alpha"]      = args.constraint_alpha
    summary["constraint_steps"]      = args.constraint_steps
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(summary)

    # Viz: three-row (Real | Background | Composite)
    if all_real:
        all_real       = torch.cat(all_real,       dim=0)
        all_composite  = torch.cat(all_composite,  dim=0)
        all_background = torch.cat(all_background, dim=0)
        rng    = np.random.default_rng(args.seed)
        idxs   = rng.choice(len(all_real), size=min(args.num_viz, len(all_real)), replace=False)
        vr     = all_real[idxs].clamp(0, 1)
        vb     = all_background[idxs].clamp(0, 1)
        vc     = all_composite[idxs].clamp(0, 1)

        n = len(idxs)
        row_labels = ["REAL\n(t+2 ground truth)", "BACKGROUND\n(t+1 cleaned)", "COMPOSITE\n(generated)"]
        row_colors = ["#1565C0", "#2E7D32", "#E65100"]  # blue, green, orange

        # One narrow labeled column on the left, then n image columns
        fig, axes_grid = plt.subplots(
            3, n + 1, figsize=(0.8 + 2 * n, 6),
            gridspec_kw={"width_ratios": [0.35] + [1] * n},
        )
        for row, (imgs, label, color) in enumerate(zip([vr, vb, vc], row_labels, row_colors)):
            # Colored label bar
            lax = axes_grid[row, 0]
            lax.set_facecolor(color)
            lax.text(0.5, 0.5, label, transform=lax.transAxes,
                     rotation=90, va="center", ha="center",
                     fontsize=8, fontweight="bold", color="white")
            lax.set_xticks([])
            lax.set_yticks([])
            for spine in lax.spines.values():
                spine.set_visible(False)
            # Image columns
            for col in range(n):
                ax = axes_grid[row, col + 1]
                ax.imshow(imgs[col].permute(1, 2, 0).numpy())
                ax.axis("off")
        fig.suptitle("SegmentVAE Rollout: Real vs Background vs Composite", fontsize=10)
        plt.tight_layout(pad=0.3)
        plt.savefig(os.path.join(args.output_dir, "segment_rollout_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {os.path.join(args.output_dir, 'segment_rollout_comparison.png')}")


if __name__ == "__main__":
    main()
