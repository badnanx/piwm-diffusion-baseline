#!/usr/bin/env python
"""Train SpriteVAE on segmented lander sprites (SegmentVAE pipeline, Stage 1/3).

Extracts lander sprites via purple color mask, pads to sprite_size×sprite_size,
and trains a small VAE to encode/decode sprite appearance.
Only visible frames are used (lander fully on-screen).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from piwm_diffusion.crop_ae import SpriteVAE, crop_vae_loss
from piwm_diffusion.data import LunarFrameDataset
from piwm_diffusion.plotting import plot_history_curves
from piwm_diffusion.sprite import extract_sprite
from piwm_diffusion.train_utils import device_from_arg, set_seed, write_json
from piwm_diffusion.viz import save_reconstruction_grid


def extract_sprites_batch(images: torch.Tensor, size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract sprites for a batch of CHW images. Returns (sprites, valid_mask)."""
    sprites, valid = [], []
    for img in images:
        s, found = extract_sprite(img, size=size)
        sprites.append(s)
        valid.append(found)
    return torch.stack(sprites).to(device), torch.tensor(valid, dtype=torch.bool)


def run_epoch(model, loader, optimizer, device, args, train):
    model.train(train)
    totals = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}
    n = 0

    for batch in tqdm(loader, desc="train" if train else "eval", leave=False):
        images = batch["image"]  # (B, 3, H, W) on CPU from loader
        sprites, valid = extract_sprites_batch(images, args.sprite_size, device)

        if valid.sum() == 0:
            continue
        sprites = sprites[valid]

        with torch.set_grad_enabled(train):
            recon, mu, logvar = model(sprites)
            losses = crop_vae_loss(recon, sprites, mu, logvar, kl_weight=args.kl_weight)
            if train:
                optimizer.zero_grad()
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        batch_n = int(valid.sum())
        n += batch_n
        for key in totals:
            totals[key] += float(losses[key].item()) * batch_n

    return {k: v / max(n, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/segment_ae")
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--sprite_size", type=int, default=32)
    parser.add_argument("--kl_weight", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--state_key", default="states")
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage_label", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = device_from_arg(args.device)

    # Always require_visible — sprite extraction needs the lander on-screen
    train_ds = LunarFrameDataset(
        args.train_dir, state_key=args.state_key,
        max_files=args.max_train_files, require_visible=True, file_seed=args.seed,
    )
    test_ds = LunarFrameDataset(
        args.test_dir, state_key=args.state_key,
        max_files=args.max_test_files, require_visible=True, file_seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)
    viz_loader   = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed + 1000))

    model = SpriteVAE(latent_dim=args.latent_dim, sprite_size=args.sprite_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("device:", device)
    print(f"sprite_size: {args.sprite_size}×{args.sprite_size}  latent_dim: {args.latent_dim}")
    print("train visible frames:", len(train_ds), "  test visible frames:", len(test_ds))

    history = []
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        prefix = f"[{args.stage_label}] " if args.stage_label else ""
        print(f"\n{prefix}Epoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, train=True)
        test_metrics  = run_epoch(model, test_loader,  optimizer, device, args, train=False)
        history.append({"epoch": epoch, "train": train_metrics, "test": test_metrics})
        print("train:", train_metrics)
        print("test: ", test_metrics)

        # Viz grid of sprite reconstructions
        with torch.no_grad():
            batch = next(iter(viz_loader))
            sprites, valid = extract_sprites_batch(batch["image"], args.sprite_size, device)
            sprites = sprites[valid][:16]
            if sprites.size(0) > 0:
                recon, _, _ = model(sprites)
                save_reconstruction_grid(
                    sprites, recon,
                    os.path.join(args.output_dir, f"recon_epoch_{epoch:03d}.png"),
                    suptitle=f"SpriteVAE reconstruction — epoch {epoch}",
                )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "latent_dim": args.latent_dim,
            "sprite_size": args.sprite_size,
            "history": history,
            "model_type": "sprite_vae",
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "latest.pt"))

        if test_metrics["loss"] < best_loss:
            best_loss = test_metrics["loss"]
            best_epoch = epoch
            stale = 0
            torch.save(checkpoint, os.path.join(args.output_dir, "best.pt"))
            if sprites.size(0) > 0:
                save_reconstruction_grid(
                    sprites, recon,
                    os.path.join(args.output_dir, "recon_best.png"),
                    suptitle=f"Best SpriteVAE reconstruction — epoch {epoch}",
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
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print("summary:", summary)


if __name__ == "__main__":
    main()
