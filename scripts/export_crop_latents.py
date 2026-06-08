#!/usr/bin/env python
"""Export CropVAE latents (mu) + theta condition for all frames."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from piwm_diffusion.crop import crop_around_state
from piwm_diffusion.crop_ae import CropVAE
from piwm_diffusion.data import LunarFrameDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_ae_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--crop_size", type=int, default=24)
    parser.add_argument("--theta_state_index", type=int, default=4,
                        help="Index in full state vector for theta (default 4)")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.crop_ae_checkpoint, map_location=device)
    ckpt_args = ckpt["args"]
    latent_dim = ckpt_args["latent_dim"]
    crop_size = ckpt_args.get("crop_size", args.crop_size)

    model = CropVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = LunarFrameDataset(args.data_dir, max_files=args.max_files)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_mu = []
    all_theta = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="exporting"):
            image = batch["image"].to(device)
            state = batch["state"].to(device)
            crops = crop_around_state(image, state, crop_size=crop_size)
            mu, _ = model.encode(crops)
            theta = state[:, args.theta_state_index]
            all_mu.append(mu.cpu().numpy())
            all_theta.append(theta.cpu().numpy())

    mu_all = np.concatenate(all_mu, axis=0)
    theta_all = np.concatenate(all_theta, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
    np.savez(args.output_npz, latents=mu_all, conditions=theta_all[:, None])
    print(f"saved {len(mu_all)} latents to {args.output_npz}")
    print(f"  latents shape: {mu_all.shape}")
    print(f"  conditions shape: {theta_all[:, None].shape}")
    print(f"  theta range: [{theta_all.min():.3f}, {theta_all.max():.3f}]")


if __name__ == "__main__":
    main()
