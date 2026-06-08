#!/usr/bin/env python
"""Train MLP DDPM on CropVAE latents conditioned on theta."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from piwm_diffusion.diffusion import ConditionalDenoiserMLP, DiffusionSchedule, ddpm_noise_prediction_loss
from piwm_diffusion.plotting import plot_history_curves
from piwm_diffusion.train_utils import set_seed, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--output_dir", default="outputs/crop_ddpm")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--diffusion_steps", type=int, default=15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage_label", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    def load_npz(path):
        d = np.load(path)
        return (
            torch.tensor(d["latents"], dtype=torch.float32),
            torch.tensor(d["conditions"], dtype=torch.float32),
        )

    train_latents, train_cond = load_npz(args.train_npz)
    val_latents, val_cond = load_npz(args.val_npz)

    latent_dim = train_latents.shape[1]
    cond_dim = train_cond.shape[1]

    print(f"device: {device}")
    print(f"latent_dim: {latent_dim}  cond_dim: {cond_dim}")
    print(f"train: {len(train_latents)}  val: {len(val_latents)}")

    train_ds = TensorDataset(train_latents, train_cond)
    val_ds = TensorDataset(val_latents, val_cond)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ConditionalDenoiserMLP(
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    schedule = DiffusionSchedule(num_steps=args.diffusion_steps, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def run_epoch(loader, train):
        model.train(train)
        total_loss = 0.0
        n = 0
        for latents_b, cond_b in tqdm(loader, desc="train" if train else "val", leave=False):
            latents_b = latents_b.to(device)
            cond_b = cond_b.to(device)
            with torch.set_grad_enabled(train):
                loss = ddpm_noise_prediction_loss(model, schedule, latents_b, cond_b)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total_loss += float(loss.item()) * len(latents_b)
            n += len(latents_b)
        return total_loss / max(n, 1)

    history = []
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        prefix = f"[{args.stage_label}] " if args.stage_label else ""
        print(f"\n{prefix}Epoch {epoch}/{args.epochs}")
        train_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(val_loader, train=False)
        history.append({"epoch": epoch, "train": {"loss": train_loss}, "test": {"loss": val_loss}})
        print(f"  train loss: {train_loss:.6f}  val loss: {val_loss:.6f}")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "schedule_steps": args.diffusion_steps,
            "epoch": epoch,
            "args": vars(args),
            "latent_dim": latent_dim,
            "cond_dim": cond_dim,
            "history": history,
            "model_type": "crop_ddpm",
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "latest.pt"))

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(checkpoint, os.path.join(args.output_dir, "best.pt"))
            print("  saved best")
        else:
            stale += 1
            print(f"  no improvement {stale}/{args.patience}")
            if stale >= args.patience:
                print("  early stopping")
                break

        write_json(os.path.join(args.output_dir, "history.json"), history)
        plot_history_curves(history, os.path.join(args.output_dir, "loss_curves.png"))

    summary = {
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "elapsed_minutes": (time.time() - start) / 60.0,
        "args": vars(args),
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print("summary:", summary)


if __name__ == "__main__":
    main()
