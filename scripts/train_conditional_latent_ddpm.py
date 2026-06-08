#!/usr/bin/env python
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from piwm_diffusion.data import LatentConditionDataset
from piwm_diffusion.diffusion import (
    ConditionalDenoiserMLP,
    DiffusionSchedule,
    ddpm_noise_prediction_loss,
    sample_latents,
)
from piwm_diffusion.plotting import plot_history_curves
from piwm_diffusion.train_utils import device_from_arg, set_seed, write_json


def train_epoch(model, schedule, loader, optimizer, device, grad_clip):
    model.train()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        x0 = batch["z_norm"].to(device)
        cond = batch["cond_norm"].to(device)
        loss = ddpm_noise_prediction_loss(model, schedule, x0, cond)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    return {"loss": float(np.mean(losses))}


@torch.no_grad()
def eval_epoch(model, schedule, loader, device, latent_dim, max_batches):
    model.eval()
    losses = []
    sample_rmses = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
        x0 = batch["z_norm"].to(device)
        cond = batch["cond_norm"].to(device)
        loss = ddpm_noise_prediction_loss(model, schedule, x0, cond)
        losses.append(float(loss.item()))

        if batch_idx < max_batches:
            sampled = sample_latents(model, schedule, cond, latent_dim)
            sample_rmses.append(float(torch.sqrt(torch.mean((sampled - x0) ** 2)).item()))

    return {
        "loss": float(np.mean(losses)),
        "sample_rmse_norm": float(np.mean(sample_rmses)) if sample_rmses else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--output_dir", default="outputs/ddpm")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--time_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_eval_sample_batches", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage_label", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = device_from_arg(args.device)

    train_ds = LatentConditionDataset(args.train_npz)
    val_ds = LatentConditionDataset(
        args.val_npz,
        latent_mean=train_ds.latent_mean,
        latent_std=train_ds.latent_std,
        cond_mean=train_ds.cond_mean,
        cond_std=train_ds.cond_std,
    )

    latent_dim = train_ds.z.shape[1]
    cond_dim = train_ds.cond.shape[1]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ConditionalDenoiserMLP(
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    schedule = DiffusionSchedule(
        num_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("device:", device)
    print("train samples:", len(train_ds), "val samples:", len(val_ds))
    print("latent_dim:", latent_dim, "cond_dim:", cond_dim)

    best_metric = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        prefix = f"[{args.stage_label}] " if args.stage_label else ""
        print(f"\n{prefix}Epoch {epoch}/{args.epochs}")
        train_metrics = train_epoch(model, schedule, train_loader, optimizer, device, args.grad_clip)
        val_metrics = eval_epoch(
            model,
            schedule,
            val_loader,
            device,
            latent_dim,
            max_batches=args.max_eval_sample_batches,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print("train:", train_metrics)
        print("val:  ", val_metrics)

        metric = val_metrics["loss"]
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "latent_dim": latent_dim,
            "cond_dim": cond_dim,
            "latent_mean": train_ds.latent_mean,
            "latent_std": train_ds.latent_std,
            "cond_mean": train_ds.cond_mean,
            "cond_std": train_ds.cond_std,
            "history": history,
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "latest.pt"))

        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            stale = 0
            torch.save(checkpoint, os.path.join(args.output_dir, "best.pt"))
            print("saved best")
        else:
            stale += 1
            print(f"no improvement {stale}/{args.patience}")
            if stale >= args.patience:
                print("early stopping")
                break

        write_json(os.path.join(args.output_dir, "history.json"), history)
        plot_history_curves(history, os.path.join(args.output_dir, "loss_curves.png"))

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_metric,
        "elapsed_minutes": (time.time() - start) / 60.0,
        "args": vars(args),
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    plot_history_curves(history, os.path.join(args.output_dir, "loss_curves.png"))
    print("summary:", summary)


if __name__ == "__main__":
    main()
