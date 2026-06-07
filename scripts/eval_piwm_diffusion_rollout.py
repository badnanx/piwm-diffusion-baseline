#!/usr/bin/env python
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from piwm_diffusion.crop import crop_around_state
from piwm_diffusion.data import LunarTripletDataset
from piwm_diffusion.diffusion import ConditionalDenoiserMLP, DiffusionSchedule, sample_latents
from piwm_diffusion.dynamics import LunarSecondOrderDynamics
from piwm_diffusion.physical import PhysicalAutoencoder
from piwm_diffusion.train_utils import device_from_arg, load_autoencoder, set_seed, write_json
from piwm_diffusion.viz import save_reconstruction_grid


@torch.no_grad()
def encode_z(autoencoder, image):
    mu, _ = autoencoder.encode(image)
    return mu


@torch.no_grad()
def encode_f(autoencoder, physical_model, image):
    z = encode_z(autoencoder, image)
    f, _ = physical_model(z)
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--physical_checkpoint", required=True)
    parser.add_argument("--dynamics_checkpoint", required=True)
    parser.add_argument("--diffusion_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/piwm_rollout_eval")
    parser.add_argument("--state_key", default="states")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_files", type=int, default=2)
    parser.add_argument("--max_triplets_per_file", type=int, default=16,
                        help="0 = no limit (use all triplets from each file)")
    parser.add_argument("--num_viz", type=int, default=8)
    parser.add_argument("--clamp_physical_dims", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_triplets_per_file == 0:
        args.max_triplets_per_file = None

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = device_from_arg(args.device)

    autoencoder, ae_ckpt = load_autoencoder(args.autoencoder_checkpoint, device)

    phys_ckpt = torch.load(args.physical_checkpoint, map_location=device, weights_only=False)
    phys_args = phys_ckpt["args"]
    physical_model = PhysicalAutoencoder(
        latent_dim=int(phys_ckpt["latent_dim"]),
        state_dim=int(phys_ckpt["state_dim"]),
        hidden_dim=int(phys_args["hidden_dim"]),
        num_layers=int(phys_args["num_layers"]),
    ).to(device)
    physical_model.load_state_dict(phys_ckpt["model_state_dict"])
    physical_model.eval()
    state_indices = list(phys_ckpt["state_indices"])

    dyn_ckpt = torch.load(args.dynamics_checkpoint, map_location=device, weights_only=False)
    dynamics = LunarSecondOrderDynamics(state_dim=int(dyn_ckpt["state_dim"])).to(device)
    dynamics.load_state_dict(dyn_ckpt["model_state_dict"])
    dynamics.eval()

    ddpm_ckpt = torch.load(args.diffusion_checkpoint, map_location=device, weights_only=False)
    ddpm_args = ddpm_ckpt["args"]
    diffusion = ConditionalDenoiserMLP(
        latent_dim=int(ddpm_ckpt["latent_dim"]),
        cond_dim=int(ddpm_ckpt["cond_dim"]),
        time_dim=int(ddpm_args["time_dim"]),
        hidden_dim=int(ddpm_args["hidden_dim"]),
        num_layers=int(ddpm_args["num_layers"]),
    ).to(device)
    diffusion.load_state_dict(ddpm_ckpt["model_state_dict"])
    diffusion.eval()
    schedule = DiffusionSchedule(
        num_steps=int(ddpm_args["diffusion_steps"]),
        beta_start=float(ddpm_args["beta_start"]),
        beta_end=float(ddpm_args["beta_end"]),
        device=device,
    )
    cond_mean = torch.from_numpy(ddpm_ckpt["cond_mean"]).to(device)
    cond_std = torch.from_numpy(ddpm_ckpt["cond_std"]).to(device)
    latent_mean = torch.from_numpy(ddpm_ckpt["latent_mean"]).to(device)
    latent_std = torch.from_numpy(ddpm_ckpt["latent_std"]).to(device)

    ds = LunarTripletDataset(
        args.data_dir,
        state_key=args.state_key,
        max_files=args.max_files,
        max_triplets_per_file=args.max_triplets_per_file,
    )
    generator = torch.Generator().manual_seed(args.seed + 2000)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)

    has_xy = 0 in state_indices and 1 in state_indices
    xi = state_indices.index(0) if has_xy else None
    yi = state_indices.index(1) if has_xy else None

    metrics = {
        "dynamics_gt_mse": [],
        "generated_image_mse": [],
        "generated_physical_vs_pred_mse": [],
        "generated_physical_vs_gt_mse": [],
        "deterministic_dp_image_mse": [],
        "generated_crop_mse": [],
        "deterministic_dp_crop_mse": [],
    }
    all_real = []
    all_generated = []

    for batch in tqdm(loader, desc="rollout"):
        image_t = batch["image_t"].to(device)
        image_t1 = batch["image_t1"].to(device)
        image_t2 = batch["image_t2"].to(device)
        gt_f2 = batch["state_t2"].to(device)[:, state_indices]
        action = batch["action_t1"].to(device)

        with torch.no_grad():
            f_t = encode_f(autoencoder, physical_model, image_t)
            f_t1 = encode_f(autoencoder, physical_model, image_t1)
            f_pred = dynamics(f_t, f_t1, action)

            cond_norm = (f_pred - cond_mean) / cond_std
            z_norm = sample_latents(diffusion, schedule, cond_norm, int(ddpm_ckpt["latent_dim"]))
            z_gen = z_norm * latent_std + latent_mean
            if args.clamp_physical_dims:
                z_gen[:, : len(state_indices)] = f_pred
            image_gen = autoencoder.decode(z_gen)

            z_dp = physical_model.decoder(f_pred)
            image_dp = autoencoder.decode(z_dp)
            f_from_gen, _ = physical_model(z_gen)

        metrics["dynamics_gt_mse"].append(F.mse_loss(f_pred, gt_f2).item())
        metrics["generated_image_mse"].append(F.mse_loss(image_gen, image_t2).item())
        metrics["generated_physical_vs_pred_mse"].append(F.mse_loss(f_from_gen, f_pred).item())
        metrics["generated_physical_vs_gt_mse"].append(F.mse_loss(f_from_gen, gt_f2).item())
        metrics["deterministic_dp_image_mse"].append(F.mse_loss(image_dp, image_t2).item())
        if has_xy:
            f_xy = torch.stack([f_pred[:, xi], f_pred[:, yi]], dim=1)
            metrics["generated_crop_mse"].append(
                F.mse_loss(crop_around_state(image_gen, f_xy), crop_around_state(image_t2, f_xy)).item()
            )
            metrics["deterministic_dp_crop_mse"].append(
                F.mse_loss(crop_around_state(image_dp, f_xy), crop_around_state(image_t2, f_xy)).item()
            )

        all_real.append(image_t2)
        all_generated.append(image_gen)

    # Randomly sample visualization frames from entire test set
    if all_real:
        all_real = torch.cat(all_real, dim=0)
        all_generated = torch.cat(all_generated, dim=0)
        indices = np.random.choice(len(all_real), size=min(args.num_viz, len(all_real)), replace=False)
        viz_real = all_real[indices]
        viz_generated = all_generated[indices]
    else:
        viz_real = None
        viz_generated = None

    summary = {
        key: float(np.mean(values)) if values else None
        for key, values in metrics.items()
    }
    summary.update(
        {
            "num_triplets": len(ds),
            "state_indices": state_indices,
            "clamp_physical_dims": bool(args.clamp_physical_dims),
        }
    )
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    if viz_real is not None and viz_generated is not None:
        save_reconstruction_grid(
            viz_real,
            viz_generated,
            os.path.join(args.output_dir, "rollout_real_vs_generated.png"),
            num_images=min(args.num_viz, viz_real.size(0)),
            suptitle="Random rollout examples: real future vs PIWM+diffusion generated",
        )
    print(summary)


if __name__ == "__main__":
    main()
