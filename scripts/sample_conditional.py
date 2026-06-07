#!/usr/bin/env python
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from piwm_diffusion.diffusion import ConditionalDenoiserMLP, DiffusionSchedule, sample_latents
from piwm_diffusion.train_utils import device_from_arg, load_autoencoder, set_seed, write_json
from piwm_diffusion.viz import save_image_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--diffusion_checkpoint", required=True)
    parser.add_argument("--conditions_npz", required=True)
    parser.add_argument("--output_dir", default="outputs/samples")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use start_index:start_index+num_samples instead of random conditions.",
    )
    parser.add_argument("--clamp_physical_dims", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = device_from_arg(args.device)

    autoencoder, ae_ckpt = load_autoencoder(args.autoencoder_checkpoint, device)
    ae_args = ae_ckpt["args"]
    state_indices = list(ae_ckpt.get("state_indices", ae_args.get("state_indices", [0, 1, 4])))

    ddpm_ckpt = torch.load(args.diffusion_checkpoint, map_location=device, weights_only=False)
    ddpm_args = ddpm_ckpt["args"]
    model = ConditionalDenoiserMLP(
        latent_dim=int(ddpm_ckpt["latent_dim"]),
        cond_dim=int(ddpm_ckpt["cond_dim"]),
        time_dim=int(ddpm_args["time_dim"]),
        hidden_dim=int(ddpm_args["hidden_dim"]),
        num_layers=int(ddpm_args["num_layers"]),
    ).to(device)
    model.load_state_dict(ddpm_ckpt["model_state_dict"])
    model.eval()
    schedule = DiffusionSchedule(
        num_steps=int(ddpm_args["diffusion_steps"]),
        beta_start=float(ddpm_args["beta_start"]),
        beta_end=float(ddpm_args["beta_end"]),
        device=device,
    )

    data = np.load(args.conditions_npz)
    all_cond = data["cond"].astype(np.float32)
    if args.sequential:
        end = args.start_index + args.num_samples
        selected_indices = np.arange(args.start_index, min(end, len(all_cond)))
    else:
        rng = np.random.default_rng(args.seed)
        replace = len(all_cond) < args.num_samples
        selected_indices = rng.choice(len(all_cond), size=args.num_samples, replace=replace)
    cond = all_cond[selected_indices]
    if len(cond) == 0:
        raise ValueError("No conditions selected")

    cond_mean = torch.from_numpy(ddpm_ckpt["cond_mean"]).to(device)
    cond_std = torch.from_numpy(ddpm_ckpt["cond_std"]).to(device)
    latent_mean = torch.from_numpy(ddpm_ckpt["latent_mean"]).to(device)
    latent_std = torch.from_numpy(ddpm_ckpt["latent_std"]).to(device)

    cond_t = torch.from_numpy(cond).to(device)
    cond_norm = (cond_t - cond_mean) / cond_std

    with torch.no_grad():
        z_norm = sample_latents(model, schedule, cond_norm, int(ddpm_ckpt["latent_dim"]))
        z = z_norm * latent_std + latent_mean
        if args.clamp_physical_dims:
            z[:, : len(state_indices)] = cond_t
        images = autoencoder.decode(z)

    titles = [", ".join(f"{v:.2f}" for v in row) for row in cond]
    grid_path = os.path.join(args.output_dir, "samples.png")
    save_image_grid(
        images,
        grid_path,
        titles=titles,
        suptitle="Random conditional latent diffusion samples",
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "samples.npz"),
        z=z.detach().cpu().numpy().astype(np.float32),
        cond=cond,
        selected_indices=selected_indices.astype(np.int64),
        state_indices=np.array(state_indices, dtype=np.int64),
    )

    summary = {
        "num_samples": int(len(cond)),
        "grid_path": grid_path,
        "conditions_npz": args.conditions_npz,
        "selected_indices": selected_indices.astype(int).tolist(),
        "clamp_physical_dims": bool(args.clamp_physical_dims),
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
