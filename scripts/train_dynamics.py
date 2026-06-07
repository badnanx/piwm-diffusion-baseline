#!/usr/bin/env python
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from piwm_diffusion.data import LunarTripletDataset, StateSpec
from piwm_diffusion.dynamics import LunarSecondOrderDynamics
from piwm_diffusion.physical import PhysicalAutoencoder
from piwm_diffusion.plotting import plot_history_curves
from piwm_diffusion.train_utils import device_from_arg, load_autoencoder, set_seed, write_json


@torch.no_grad()
def encoded_physical(autoencoder, physical_model, image):
    mu, _ = autoencoder.encode(image)
    state, _ = physical_model(mu)
    return state


def run_epoch(autoencoder, physical_model, dynamics, loader, optimizer, device, args, train):
    dynamics.train(train)
    totals = {"loss": 0.0, "encoded_target_loss": 0.0, "gt_target_loss": 0.0}
    n = 0

    for batch in tqdm(loader, desc="train" if train else "eval", leave=False):
        image_t = batch["image_t"].to(device)
        image_t1 = batch["image_t1"].to(device)
        image_t2 = batch["image_t2"].to(device)
        target_gt = batch["state_t2"].to(device)[:, args.state_indices]
        action = batch["action_t1"].to(device)

        f_t = encoded_physical(autoencoder, physical_model, image_t)
        f_t1 = encoded_physical(autoencoder, physical_model, image_t1)
        f_t2_encoded = encoded_physical(autoencoder, physical_model, image_t2)

        with torch.set_grad_enabled(train):
            pred = dynamics(f_t, f_t1, action)
            encoded_target_loss = F.mse_loss(pred, f_t2_encoded, reduction="mean")
            gt_target_loss = F.mse_loss(pred, target_gt, reduction="mean")
            loss = args.encoded_target_weight * encoded_target_loss + args.gt_target_weight * gt_target_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dynamics.parameters(), args.grad_clip)
                optimizer.step()

        batch_n = image_t.size(0)
        n += batch_n
        totals["loss"] += float(loss.item()) * batch_n
        totals["encoded_target_loss"] += float(encoded_target_loss.item()) * batch_n
        totals["gt_target_loss"] += float(gt_target_loss.item()) * batch_n

    return {key: value / max(n, 1) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--physical_checkpoint", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/dynamics")
    parser.add_argument("--state_key", default="states")
    parser.add_argument("--encoded_target_weight", type=float, default=1.0)
    parser.add_argument("--gt_target_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--max_triplets_per_file", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = device_from_arg(args.device)

    autoencoder, ae_ckpt = load_autoencoder(args.autoencoder_checkpoint, device)
    ae_args = ae_ckpt["args"]

    phys_ckpt = torch.load(args.physical_checkpoint, map_location=device, weights_only=False)
    phys_args = phys_ckpt["args"]
    state_indices = list(phys_ckpt["state_indices"])
    args.state_indices = state_indices
    physical_model = PhysicalAutoencoder(
        latent_dim=int(phys_ckpt["latent_dim"]),
        state_dim=int(phys_ckpt["state_dim"]),
        hidden_dim=int(phys_args["hidden_dim"]),
        num_layers=int(phys_args["num_layers"]),
    ).to(device)
    physical_model.load_state_dict(phys_ckpt["model_state_dict"])
    physical_model.eval()
    for param in physical_model.parameters():
        param.requires_grad_(False)

    dynamics = LunarSecondOrderDynamics(state_dim=len(state_indices)).to(device)
    optimizer = torch.optim.AdamW(dynamics.parameters(), lr=args.lr)

    train_ds = LunarTripletDataset(
        args.train_dir,
        state_key=args.state_key,
        max_files=args.max_train_files,
        max_triplets_per_file=args.max_triplets_per_file,
    )
    test_ds = LunarTripletDataset(
        args.test_dir,
        state_key=args.state_key,
        max_files=args.max_test_files,
        max_triplets_per_file=args.max_triplets_per_file,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    state_spec = StateSpec.from_indices(state_indices)
    print("device:", device)
    print("physical state:", list(zip(state_spec.indices, state_spec.names)))
    print("train triplets:", len(train_ds), "test triplets:", len(test_ds))

    history = []
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(autoencoder, physical_model, dynamics, train_loader, optimizer, device, args, train=True)
        test_metrics = run_epoch(autoencoder, physical_model, dynamics, test_loader, optimizer, device, args, train=False)
        row = {"epoch": epoch, "train": train_metrics, "test": test_metrics}
        history.append(row)
        print("train:", train_metrics)
        print("test: ", test_metrics)

        checkpoint = {
            "model_state_dict": dynamics.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "state_dim": len(state_indices),
            "state_indices": state_indices,
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
        "learned_parameters": {
            "main_power": float(dynamics.main_power.detach().cpu().item()),
            "side_power": float(dynamics.side_power.detach().cpu().item()),
            "angular_power": float(dynamics.angular_power.detach().cpu().item()),
            "gravity": float(dynamics.gravity.detach().cpu().item()),
        },
        "elapsed_minutes": (time.time() - start) / 60.0,
        "args": vars(args),
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    plot_history_curves(history, os.path.join(args.output_dir, "loss_curves.png"))
    print("summary:", summary)


if __name__ == "__main__":
    main()
