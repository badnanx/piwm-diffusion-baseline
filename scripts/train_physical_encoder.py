#!/usr/bin/env python
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from piwm_diffusion.data import LunarFrameDataset, StateSpec
from piwm_diffusion.physical import PhysicalAutoencoder, physical_autoencoder_loss
from piwm_diffusion.plotting import plot_history_curves
from piwm_diffusion.train_utils import device_from_arg, load_autoencoder, set_seed, write_json


def encode_latent(autoencoder, image):
    with torch.no_grad():
        mu, _ = autoencoder.encode(image)
    return mu


def run_epoch(autoencoder, model, loader, optimizer, device, args, train):
    model.train(train)
    totals = {"loss": 0.0, "state_loss": 0.0, "latent_recon_loss": 0.0}
    n = 0

    for batch in tqdm(loader, desc="train" if train else "eval", leave=False):
        image = batch["image"].to(device)
        state = batch["state"].to(device)
        z = encode_latent(autoencoder, image)
        target_state = state[:, args.state_indices]

        with torch.set_grad_enabled(train):
            pred_state, z_recon = model(z)
            losses = physical_autoencoder_loss(
                pred_state=pred_state,
                target_state=target_state,
                z_recon=z_recon,
                z=z,
                state_weight=args.state_weight,
                latent_recon_weight=args.latent_recon_weight,
            )
            if train:
                optimizer.zero_grad()
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        batch_n = image.size(0)
        n += batch_n
        for key in totals:
            totals[key] += float(losses[key].item()) * batch_n

    return {key: value / max(n, 1) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/physical_encoder")
    parser.add_argument("--state_indices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--state_key", default="states")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--state_weight", type=float, default=1.0)
    parser.add_argument("--latent_recon_weight", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--max_frames_per_file", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage_label", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = device_from_arg(args.device)
    state_spec = StateSpec.from_indices(args.state_indices)

    autoencoder, ae_ckpt = load_autoencoder(args.autoencoder_checkpoint, device)
    latent_dim = int(ae_ckpt["args"]["latent_dim"])

    train_ds = LunarFrameDataset(
        args.train_dir,
        state_key=args.state_key,
        max_files=args.max_train_files,
        max_frames_per_file=args.max_frames_per_file,
    )
    test_ds = LunarFrameDataset(
        args.test_dir,
        state_key=args.state_key,
        max_files=args.max_test_files,
        max_frames_per_file=args.max_frames_per_file,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PhysicalAutoencoder(
        latent_dim=latent_dim,
        state_dim=len(args.state_indices),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("device:", device)
    print("latent_dim:", latent_dim)
    print("physical state:", list(zip(state_spec.indices, state_spec.names)))
    print("train frames:", len(train_ds), "test frames:", len(test_ds))

    history = []
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        prefix = f"[{args.stage_label}] " if args.stage_label else ""
        print(f"\n{prefix}Epoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(autoencoder, model, train_loader, optimizer, device, args, train=True)
        test_metrics = run_epoch(autoencoder, model, test_loader, optimizer, device, args, train=False)
        row = {"epoch": epoch, "train": train_metrics, "test": test_metrics}
        history.append(row)
        print("train:", train_metrics)
        print("test: ", test_metrics)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "latent_dim": latent_dim,
            "state_dim": len(args.state_indices),
            "state_indices": args.state_indices,
            "state_names": state_spec.names,
            "history": history,
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "latest.pt"))

        if test_metrics["loss"] < best_loss:
            best_loss = test_metrics["loss"]
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
        "best_loss": best_loss,
        "elapsed_minutes": (time.time() - start) / 60.0,
        "args": vars(args),
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    plot_history_curves(history, os.path.join(args.output_dir, "loss_curves.png"))
    print("summary:", summary)


if __name__ == "__main__":
    main()
