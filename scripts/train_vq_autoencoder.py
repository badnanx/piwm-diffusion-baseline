#!/usr/bin/env python
"""Train VQ-VAE for Lunar Lander with crop loss support."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from piwm_diffusion.vq_autoencoder import PiwmVQVAE, vq_vae_loss
from piwm_diffusion.crop import state_guided_crop_mse
from piwm_diffusion.data import LunarFrameDataset, StateSpec
from piwm_diffusion.plotting import plot_history_curves
from piwm_diffusion.train_utils import device_from_arg, set_seed, write_json
from piwm_diffusion.viz import save_reconstruction_grid, save_reconstruction_grid_boxed


def run_epoch(model, loader, optimizer, device, args, train):
    model.train(train)
    totals = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "vq_loss": 0.0,
        "crop_loss": 0.0,
    }
    n = 0

    for batch in tqdm(loader, desc="train" if train else "eval", leave=False):
        image = batch["image"].to(device)
        state = batch["state"].to(device)

        with torch.set_grad_enabled(train):
            recon, vq_loss, perplexity = model(image)
            losses = vq_vae_loss(
                recon=recon,
                image=image,
                vq_loss=vq_loss,
                state=state,
                state_indices=args.state_indices,
                state_weight=0.0,  # No direct state supervision for VQ-VAE
                recon_weight=args.recon_weight,
            )
            crop_loss = (
                state_guided_crop_mse(
                    pred_images=recon,
                    target_images=image,
                    states=state,
                    crop_size=args.crop_size,
                )
                if args.crop_weight > 0.0
                else torch.zeros((), device=device)
            )
            losses["loss"] = losses["loss"] + args.crop_weight * crop_loss
            losses["crop_loss"] = crop_loss
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
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/vq_autoencoder")
    parser.add_argument("--latent_dim", type=int, default=48)
    parser.add_argument("--num_codebook_vectors", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.25, help="VQ commitment cost")
    parser.add_argument("--state_indices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--state_key", default="states")
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--crop_weight", type=float, default=0.0)
    parser.add_argument("--crop_size", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--max_frames_per_file", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = device_from_arg(args.device)
    state_spec = StateSpec.from_indices(args.state_indices)

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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    viz_generator = torch.Generator().manual_seed(args.seed + 1000)
    viz_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=viz_generator,
    )

    model = PiwmVQVAE(
        latent_dim=args.latent_dim,
        num_codebook_vectors=args.num_codebook_vectors,
        beta=args.beta,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    start = time.time()

    print("device:", device)
    print("latent_dim:", args.latent_dim)
    print("num_codebook_vectors:", args.num_codebook_vectors)
    print("beta:", args.beta)
    print("crop_weight:", args.crop_weight)
    print("crop_size:", args.crop_size)
    print("train frames:", len(train_ds), "test frames:", len(test_ds))

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, train=True)
        test_metrics = run_epoch(model, test_loader, optimizer, device, args, train=False)
        row = {"epoch": epoch, "train": train_metrics, "test": test_metrics}
        history.append(row)
        print("train:", train_metrics)
        print("test: ", test_metrics)

        with torch.no_grad():
            batch = next(iter(viz_loader))
            image = batch["image"].to(device)
            recon, vq_loss, perplexity = model(image)
            epoch_grid_path = os.path.join(args.output_dir, f"recon_epoch_{epoch:03d}.png")
            save_reconstruction_grid(
                image,
                recon,
                epoch_grid_path,
                suptitle=f"VQ-VAE reconstruction - epoch {epoch}",
            )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "latent_dim": args.latent_dim,
            "state_indices": args.state_indices,
            "state_names": state_spec.names,
            "history": history,
            "model_type": "vq_vae",
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "latest.pt"))

        if test_metrics["loss"] < best_loss:
            best_loss = test_metrics["loss"]
            best_epoch = epoch
            stale = 0
            torch.save(checkpoint, os.path.join(args.output_dir, "best.pt"))
            save_reconstruction_grid(
                image,
                recon,
                os.path.join(args.output_dir, "recon_best.png"),
                suptitle=f"Best VQ-VAE reconstruction - epoch {epoch}",
            )
            if args.crop_weight > 0.0:
                save_reconstruction_grid_boxed(
                    image,
                    recon,
                    batch["state"],
                    os.path.join(args.output_dir, "recon_best_boxed.png"),
                    crop_size=args.crop_size,
                    suptitle=f"Best VQ-VAE reconstruction with cyan crop boxes - epoch {epoch}",
                )
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
        "model_type": "vq_vae",
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    plot_history_curves(history, os.path.join(args.output_dir, "loss_curves.png"))
    print("summary:", summary)


if __name__ == "__main__":
    main()
