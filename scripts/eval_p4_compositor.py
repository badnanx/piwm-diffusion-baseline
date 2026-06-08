#!/usr/bin/env python
"""
P4 rollout evaluation: compose generated lander crop onto background at predicted (x,y).

Pipeline per step:
  1. background = image_t  (copy from current frame)
  2. f_t  = physical_model.encoder(encode(image_t))
  3. f_t1 = physical_model.encoder(encode(image_t1))
  4. f_t2 = dynamics(f_t, f_t1, action_t1)
  5. crop_latent = crop_ddpm.sample(cond=f_t2[theta_idx])
  6. crop_img = crop_vae.decode(crop_latent)
  7. paste crop_img onto background at pixel coords from f_t2 (x, y)
"""
import argparse
import json
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

from piwm_diffusion.crop import crop_around_state, state_xy_to_pixel
from piwm_diffusion.crop_ae import CropVAE
from piwm_diffusion.data import LunarTripletDataset
from piwm_diffusion.diffusion import ConditionalDenoiserMLP, DiffusionSchedule, sample_latents
from piwm_diffusion.dynamics import LunarSecondOrderDynamics
from piwm_diffusion.physical import PhysicalAutoencoder
from piwm_diffusion.train_utils import device_from_arg, load_autoencoder, set_seed


def load_crop_ae(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CropVAE(latent_dim=ckpt["args"]["latent_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def load_crop_ddpm(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt["args"]
    model = ConditionalDenoiserMLP(
        latent_dim=ckpt["latent_dim"],
        cond_dim=ckpt["cond_dim"],
        hidden_dim=ckpt_args.get("hidden_dim", 128),
        num_layers=ckpt_args.get("num_layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    schedule = DiffusionSchedule(
        num_steps=ckpt.get("schedule_steps", ckpt_args.get("diffusion_steps", 15)),
        device=device,
    )
    return model, schedule, ckpt


def lander_pixel_mask(img, lander_threshold=0.08):
    """
    Boolean mask of lander pixels (purple body + red/orange thruster fire).
    Purple: blue clearly dominates red and green (rejects white/grey terrain).
    Red/orange: red clearly dominates blue and green.
    img: (3, H, W) float tensor in [0, 1]. Returns (H, W) bool tensor.
    """
    r, g, b = img[0], img[1], img[2]
    purple = (b > lander_threshold) & (b > g + 0.05) & (b > r + 0.05)
    fire   = (r > lander_threshold) & (r > b + 0.05) & (r > g * 0.8)
    return purple | fire


def erase_lander(image, state, crop_size, lander_threshold=0.08):
    """
    Black out lander pixels in a region around the known state position.
    image: (3, H, W). Returns (3, H, W) with lander zeroed out.
    """
    _, H, W = image.shape
    fake_state = torch.zeros(1, 8, device=image.device)
    fake_state[0, 0] = state[0]
    fake_state[0, 1] = state[1]
    px, py = state_xy_to_pixel(fake_state, H, W)
    px, py = int(round(float(px[0]))), int(round(float(py[0])))

    # Search window is 2x crop_size to catch the full lander including legs
    half = crop_size
    x0, y0 = max(0, px - half), max(0, py - half)
    x1, y1 = min(W, px + half), min(H, py + half)

    result = image.clone()
    region = result[:, y0:y1, x0:x1]
    mask = lander_pixel_mask(region, lander_threshold)
    result[:, y0:y1, x0:x1] = region * (~mask).float()
    return result


def paste_crop(background, crop_img, px, py, crop_size, lander_threshold=0.08):
    """
    Paste crop_img onto background at pixel (px, py).
    Uses color mask (purple + fire) to select lander pixels from the crop.
    """
    _, H, W = background.shape
    half = crop_size // 2
    x0 = int(round(float(px))) - half
    y0 = int(round(float(py))) - half
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    cx0 = max(0, -x0)
    cy0 = max(0, -y0)
    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(W, x1)
    iy1 = min(H, y1)
    cx1 = cx0 + (ix1 - ix0)
    cy1 = cy0 + (iy1 - iy0)

    if ix1 <= ix0 or iy1 <= iy0:
        return background.clone()

    composite = background.clone()
    crop_region = crop_img[:, cy0:cy1, cx0:cx1]
    mask = lander_pixel_mask(crop_region, lander_threshold).float().unsqueeze(0)
    composite[:, iy0:iy1, ix0:ix1] = (
        mask * crop_region + (1.0 - mask) * background[:, iy0:iy1, ix0:ix1]
    )
    return composite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--physical_checkpoint", required=True)
    parser.add_argument("--dynamics_checkpoint", required=True)
    parser.add_argument("--crop_ae_checkpoint", required=True)
    parser.add_argument("--crop_ddpm_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/p4_eval")
    parser.add_argument("--state_indices", type=int, nargs="+", default=[0, 1, 4])
    parser.add_argument("--theta_f_index", type=int, default=2,
                        help="Index of theta in f vector (default 2 for state_indices=[0,1,4])")
    parser.add_argument("--crop_size", type=int, default=24)
    parser.add_argument("--lander_threshold", type=float, default=0.08)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_triplets_per_file", type=int, default=0,
                        help="0 = no limit")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_viz", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = device_from_arg(args.device)
    max_triplets = None if args.max_triplets_per_file == 0 else args.max_triplets_per_file

    # Load all models
    autoencoder, ae_ckpt = load_autoencoder(args.autoencoder_checkpoint, device)
    latent_dim = int(ae_ckpt["args"]["latent_dim"])
    state_dim = len(args.state_indices)

    phys_ckpt = torch.load(args.physical_checkpoint, map_location=device)
    phys_args = phys_ckpt["args"]
    physical_model = PhysicalAutoencoder(
        latent_dim=latent_dim,
        state_dim=state_dim,
        hidden_dim=phys_args.get("hidden_dim", 128),
        num_layers=phys_args.get("num_layers", 2),
    ).to(device)
    physical_model.load_state_dict(phys_ckpt["model_state_dict"])
    physical_model.eval()
    for p in physical_model.parameters():
        p.requires_grad_(False)

    dyn_ckpt = torch.load(args.dynamics_checkpoint, map_location=device)
    dynamics = LunarSecondOrderDynamics(state_dim=state_dim).to(device)
    dynamics.load_state_dict(dyn_ckpt["model_state_dict"])
    dynamics.eval()
    for p in dynamics.parameters():
        p.requires_grad_(False)

    crop_vae, _ = load_crop_ae(args.crop_ae_checkpoint, device)
    crop_ddpm, crop_schedule, _ = load_crop_ddpm(args.crop_ddpm_checkpoint, device)

    ds = LunarTripletDataset(
        args.data_dir,
        max_files=args.max_files,
        max_triplets_per_file=max_triplets,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # Separate shuffled loader for viz so samples aren't all from episode starts
    viz_loader = DataLoader(
        ds, batch_size=args.num_viz, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed + 99),
    )

    all_crop_mse = []

    def run_batch(batch):
        image_t  = batch["image_t"].to(device)
        image_t1 = batch["image_t1"].to(device)
        image_t2 = batch["image_t2"].to(device)
        state_t  = batch["state_t"].to(device)
        state_t2 = batch["state_t2"].to(device)
        action   = batch["action_t1"].to(device)
        B = image_t.size(0)
        _, _, H, W = image_t.shape

        z_t,  _ = autoencoder.encode(image_t)
        z_t1, _ = autoencoder.encode(image_t1)
        f_t,  _ = physical_model(z_t)
        f_t1, _ = physical_model(z_t1)

        f_t2_pred = dynamics(f_t, f_t1, action)

        x_pred    = f_t2_pred[:, 0]
        y_pred    = f_t2_pred[:, 1]
        theta_pred = f_t2_pred[:, args.theta_f_index]

        cond = theta_pred.unsqueeze(1)
        crop_latent = sample_latents(crop_ddpm, crop_schedule, cond, crop_vae.latent_dim)
        crop_imgs = crop_vae.decode(crop_latent)

        fake_state = torch.zeros(B, 8, device=device)
        fake_state[:, 0] = x_pred
        fake_state[:, 1] = y_pred
        px_batch, py_batch = state_xy_to_pixel(fake_state, H, W)

        composites = []
        for i in range(B):
            bg = erase_lander(image_t[i], state_t[i], args.crop_size, args.lander_threshold)
            comp = paste_crop(
                bg, crop_imgs[i],
                px_batch[i], py_batch[i],
                args.crop_size, args.lander_threshold,
            )
            composites.append(comp)
        composites = torch.stack(composites)

        comp_crop = crop_around_state(composites, state_t2, crop_size=args.crop_size)
        real_crop = crop_around_state(image_t2, state_t2, crop_size=args.crop_size)
        crop_mse = F.mse_loss(comp_crop, real_crop, reduction="none").mean(dim=[1, 2, 3])

        return {
            "image_t": image_t, "image_t2": image_t2, "state_t2": state_t2,
            "composites": composites, "crop_imgs": crop_imgs,
            "comp_crop": comp_crop, "real_crop": real_crop,
            "crop_mse": crop_mse,
        }

    # Full metrics pass (shuffled=False for reproducibility)
    with torch.no_grad():
        for batch in loader:
            out = run_batch(batch)
            all_crop_mse.extend(out["crop_mse"].cpu().tolist())

    # Viz pass — one shuffled batch
    with torch.no_grad():
        viz_batch = next(iter(viz_loader))
        viz_out = run_batch(viz_batch)

    mean_crop_mse = float(np.mean(all_crop_mse))
    print(f"crop_mse mean: {mean_crop_mse:.6f}  n={len(all_crop_mse)}")

    # 1. Full frame: real t2 vs composite
    viz_rows = [
        {
            "real": viz_out["image_t2"][i].cpu(),
            "generated": viz_out["composites"][i].cpu(),
            "crop_mse": viz_out["crop_mse"][i].item(),
        }
        for i in range(viz_out["image_t2"].size(0))
    ]
    _save_comparison_grid(
        viz_rows,
        os.path.join(args.output_dir, "p4_rollout_real_vs_generated.png"),
    )

    # 2. Crop-only grid: generated crop vs real crop at true next position
    _save_crop_grid(
        viz_out["comp_crop"].cpu(),
        viz_out["real_crop"].cpu(),
        os.path.join(args.output_dir, "p4_crop_generated_vs_real.png"),
    )

    # 3. Component view: background | generated crop | composite | real t2
    _save_component_grid(
        viz_out["image_t"].cpu(),
        viz_out["crop_imgs"].cpu(),
        viz_out["composites"].cpu(),
        viz_out["image_t2"].cpu(),
        os.path.join(args.output_dir, "p4_components.png"),
    )

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"mean_crop_mse": mean_crop_mse, "n": len(all_crop_mse)}, f, indent=2)


def _save_crop_grid(gen_crops, real_crops, path, ncols=8):
    """Side-by-side 24x24 crop comparison: generated (top) vs real (bottom)."""
    n = gen_crops.size(0)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(ncols * 1.5, nrows * 3))
    axes = np.array(axes).reshape(nrows * 2, ncols)
    for idx in range(n):
        r = (idx // ncols) * 2
        c = idx % ncols
        axes[r,     c].imshow(gen_crops[idx].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[r,     c].set_title("gen", fontsize=6)
        axes[r,     c].axis("off")
        axes[r + 1, c].imshow(real_crops[idx].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[r + 1, c].set_title("real", fontsize=6)
        axes[r + 1, c].axis("off")
    for idx in range(n, nrows * ncols):
        r = (idx // ncols) * 2
        c = idx % ncols
        axes[r, c].axis("off")
        axes[r + 1, c].axis("off")
    plt.suptitle("CropDDPM generated (top) vs real (bottom) at true lander position", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def _save_component_grid(backgrounds, crop_imgs, composites, reals, path, ncols=4):
    """4-panel per example: background | generated crop (upscaled) | composite | real t2."""
    n = backgrounds.size(0)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows * 4, ncols, figsize=(ncols * 3, nrows * 12))
    axes = np.array(axes).reshape(nrows * 4, ncols)
    H, W = backgrounds.shape[2], backgrounds.shape[3]
    for idx in range(n):
        r = (idx // ncols) * 4
        c = idx % ncols
        # Upscale crop to full frame size for display
        crop_up = F.interpolate(crop_imgs[idx:idx+1], size=(H, W), mode="nearest")[0]
        axes[r,     c].imshow(backgrounds[idx].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[r,     c].set_title("background (t)", fontsize=6)
        axes[r,     c].axis("off")
        axes[r + 1, c].imshow(crop_up.permute(1, 2, 0).clamp(0, 1).numpy())
        axes[r + 1, c].set_title("generated crop", fontsize=6)
        axes[r + 1, c].axis("off")
        axes[r + 2, c].imshow(composites[idx].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[r + 2, c].set_title("composite", fontsize=6)
        axes[r + 2, c].axis("off")
        axes[r + 3, c].imshow(reals[idx].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[r + 3, c].set_title("real (t+2)", fontsize=6)
        axes[r + 3, c].axis("off")
    for idx in range(n, nrows * ncols):
        r = (idx // ncols) * 4
        c = idx % ncols
        for dr in range(4):
            axes[r + dr, c].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def _save_comparison_grid(viz_rows, path, ncols=4):
    n = len(viz_rows)
    if n == 0:
        return
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(ncols * 3, nrows * 6))
    axes = np.array(axes).reshape(nrows * 2, ncols)

    for idx, row in enumerate(viz_rows):
        r = (idx // ncols) * 2
        c = idx % ncols
        real_np = row["real"].permute(1, 2, 0).clamp(0, 1).numpy()
        gen_np  = row["generated"].permute(1, 2, 0).clamp(0, 1).numpy()
        axes[r,     c].imshow(real_np)
        axes[r,     c].set_title(f"real {idx}", fontsize=7)
        axes[r,     c].axis("off")
        axes[r + 1, c].imshow(gen_np)
        axes[r + 1, c].set_title(f"gen {idx} mse={row['crop_mse']:.4f}", fontsize=7)
        axes[r + 1, c].axis("off")

    for idx in range(n, nrows * ncols):
        r = (idx // ncols) * 2
        c = idx % ncols
        axes[r,     c].axis("off")
        axes[r + 1, c].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
