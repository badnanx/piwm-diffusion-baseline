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


def paste_crop(background, crop_img, px, py, crop_size, lander_threshold=0.08):
    """
    Paste crop_img onto a copy of background at pixel (px, py).
    Lander pixels selected by max(RGB) > threshold.

    Args:
        background: (3, H, W) float tensor
        crop_img:   (3, crop_size, crop_size) float tensor
        px, py:     float pixel center coords
        crop_size:  int
        lander_threshold: float

    Returns:
        composite: (3, H, W) float tensor
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
    mask = (crop_region.max(dim=0).values > lander_threshold).float().unsqueeze(0)
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

    all_crop_mse = []
    viz_rows = []

    with torch.no_grad():
        for batch in loader:
            image_t  = batch["image_t"].to(device)
            image_t1 = batch["image_t1"].to(device)
            image_t2 = batch["image_t2"].to(device)
            state_t2 = batch["state_t2"].to(device)
            action   = batch["action_t1"].to(device)
            B = image_t.size(0)
            _, _, H, W = image_t.shape

            # Encode to latents then physical features
            z_t,  _ = autoencoder.encode(image_t)
            z_t1, _ = autoencoder.encode(image_t1)
            f_t,  _ = physical_model(z_t)
            f_t1, _ = physical_model(z_t1)

            # Predict next physical state
            f_t2_pred = dynamics(f_t, f_t1, action)

            # x, y are the first two dims of f (state_indices[0] and [1])
            x_pred = f_t2_pred[:, 0]
            y_pred = f_t2_pred[:, 1]
            theta_pred = f_t2_pred[:, args.theta_f_index]

            # Sample crop from DDPM conditioned on theta
            cond = theta_pred.unsqueeze(1)
            crop_latent = sample_latents(crop_ddpm, crop_schedule, cond, crop_vae.latent_dim)
            crop_imgs = crop_vae.decode(crop_latent)

            # Convert predicted (x, y) to pixel coordinates
            fake_state = torch.zeros(B, 8, device=device)
            fake_state[:, 0] = x_pred
            fake_state[:, 1] = y_pred
            px_batch, py_batch = state_xy_to_pixel(fake_state, H, W)

            # Compose: paste crop onto background (image_t)
            composites = []
            for i in range(B):
                comp = paste_crop(
                    image_t[i], crop_imgs[i],
                    px_batch[i], py_batch[i],
                    args.crop_size, args.lander_threshold,
                )
                composites.append(comp)
            composites = torch.stack(composites)

            # Crop MSE between composite and real next frame, both cropped around true next state
            comp_crop = crop_around_state(composites, state_t2, crop_size=args.crop_size)
            real_crop = crop_around_state(image_t2, state_t2, crop_size=args.crop_size)
            crop_mse = F.mse_loss(comp_crop, real_crop, reduction="none").mean(dim=[1, 2, 3])
            all_crop_mse.extend(crop_mse.cpu().tolist())

            if len(viz_rows) < args.num_viz:
                for i in range(min(B, args.num_viz - len(viz_rows))):
                    viz_rows.append({
                        "real": image_t2[i].cpu(),
                        "generated": composites[i].cpu(),
                        "crop_mse": crop_mse[i].item(),
                    })

    mean_crop_mse = float(np.mean(all_crop_mse))
    print(f"crop_mse mean: {mean_crop_mse:.6f}  n={len(all_crop_mse)}")

    _save_comparison_grid(
        viz_rows,
        os.path.join(args.output_dir, "p4_rollout_real_vs_generated.png"),
    )

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"mean_crop_mse": mean_crop_mse, "n": len(all_crop_mse)}, f, indent=2)


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
