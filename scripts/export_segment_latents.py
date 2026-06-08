#!/usr/bin/env python
"""Export SpriteVAE latents + theta condition for all visible frames (SegmentVAE Stage 2/3).

Output npz uses z/cond keys compatible with train_conditional_latent_ddpm.py.
Conditioning is theta only (state index 4) — sprite appearance depends on angle,
not on x/y position (which determines where to paste, not what it looks like).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from piwm_diffusion.crop_ae import SpriteVAE
from piwm_diffusion.data import LunarFrameDataset
from piwm_diffusion.sprite import extract_sprite
from piwm_diffusion.train_utils import device_from_arg, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="SpriteVAE checkpoint (best.pt)")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--theta_state_index", type=int, default=4,
                        help="Index of theta in the raw 8D state vector (default 4)")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    latent_dim  = int(ckpt["latent_dim"])
    sprite_size = int(ckpt.get("sprite_size", 32))

    model = SpriteVAE(latent_dim=latent_dim, sprite_size=sprite_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = LunarFrameDataset(args.data_dir, max_files=args.max_files,
                           require_visible=True, file_seed=42)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_z, all_theta = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="export"):
            images = batch["image"]   # CPU
            states = batch["state"]   # CPU

            sprites, valid = [], []
            for img in images:
                s, found = extract_sprite(img, size=sprite_size)
                sprites.append(s)
                valid.append(found)
            sprites = torch.stack(sprites).to(device)
            valid   = torch.tensor(valid, dtype=torch.bool)

            if valid.sum() == 0:
                continue

            sprites = sprites[valid]
            states_v = states[valid]

            mu, _ = model.encode(sprites)
            theta = states_v[:, args.theta_state_index]

            all_z.append(mu.cpu().numpy().astype(np.float32))
            all_theta.append(theta.numpy().astype(np.float32))

    z     = np.concatenate(all_z,     axis=0)
    theta = np.concatenate(all_theta, axis=0)
    cond  = theta[:, None]   # (N, 1) — compatible with LatentConditionDataset

    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
    np.savez_compressed(args.output_npz, z=z, cond=cond)

    summary = {
        "output_npz": args.output_npz,
        "num_samples": int(len(z)),
        "latent_dim": int(z.shape[1]),
        "cond_dim": int(cond.shape[1]),
        "theta_range": [float(theta.min()), float(theta.max())],
        "sprite_size": sprite_size,
    }
    write_json(os.path.splitext(args.output_npz)[0] + "_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
