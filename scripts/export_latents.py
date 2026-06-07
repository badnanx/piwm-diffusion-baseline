#!/usr/bin/env python
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from piwm_diffusion.autoencoder import PiwmConvVAE
from piwm_diffusion.vq_autoencoder import PiwmVQVAE
from piwm_diffusion.data import LunarFrameDataset
from piwm_diffusion.train_utils import device_from_arg, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--state_key", default="states")
    parser.add_argument(
        "--state_indices",
        type=int,
        nargs="+",
        default=None,
        help="Condition state indices to export. Defaults to checkpoint state indices.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_frames_per_file", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt["args"]
    latent_dim = int(model_args["latent_dim"])
    model_type = ckpt.get("model_type", "continuous")  # Default to continuous for backwards compat

    if args.state_indices is None:
        state_indices = list(ckpt.get("state_indices", model_args.get("state_indices", [0, 1, 4])))
    else:
        state_indices = list(args.state_indices)

    # Load appropriate model type
    if model_type == "vq_vae":
        model = PiwmVQVAE(
            latent_dim=latent_dim,
            num_codebook_vectors=int(model_args.get("num_codebook_vectors", 512)),
            beta=float(model_args.get("beta", 0.25)),
        ).to(device)
    else:
        model = PiwmConvVAE(latent_dim=latent_dim).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = LunarFrameDataset(
        args.data_dir,
        state_key=args.state_key,
        max_files=args.max_files,
        max_frames_per_file=args.max_frames_per_file,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    zs = []
    states = []
    conds = []
    actions = []
    recon_mses = []

    for batch in tqdm(loader, desc="export"):
        image = batch["image"].to(device)
        state = batch["state"].to(device)
        with torch.no_grad():
            mu, _ = model.encode(image)
            recon = model.decode(mu)
            mse = torch.mean((recon - image) ** 2, dim=(1, 2, 3))

        zs.append(mu.cpu().numpy().astype(np.float32))
        states.append(state.cpu().numpy().astype(np.float32))
        conds.append(state[:, state_indices].cpu().numpy().astype(np.float32))
        actions.append(batch["action"].numpy().astype(np.int64))
        recon_mses.append(mse.cpu().numpy().astype(np.float32))

    z = np.concatenate(zs, axis=0)
    state = np.concatenate(states, axis=0)
    cond = np.concatenate(conds, axis=0)
    action = np.concatenate(actions, axis=0)
    recon_mse = np.concatenate(recon_mses, axis=0)

    os.makedirs(os.path.dirname(args.output_npz), exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        z=z,
        states=state,
        cond=cond,
        actions=action,
        recon_mse=recon_mse,
        state_indices=np.array(state_indices, dtype=np.int64),
    )

    summary = {
        "output_npz": args.output_npz,
        "num_samples": int(len(z)),
        "latent_dim": int(z.shape[1]),
        "cond_dim": int(cond.shape[1]),
        "state_indices": state_indices,
        "mean_recon_mse": float(recon_mse.mean()),
    }
    write_json(os.path.splitext(args.output_npz)[0] + "_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
