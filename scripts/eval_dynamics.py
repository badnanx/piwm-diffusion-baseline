#!/usr/bin/env python
"""
Evaluate the dynamics model visually.

Produces two outputs:
  scatter.png  -- pred vs true scatter plot for each state variable (f_t+2)
  overlay.png  -- real frames with true (green) and predicted (red) lander position dots
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch.utils.data import DataLoader

from piwm_diffusion.crop import state_xy_to_pixel
from piwm_diffusion.data import LunarTripletDataset, StateSpec
from piwm_diffusion.dynamics import LunarSecondOrderDynamics
from piwm_diffusion.physical import PhysicalAutoencoder
from piwm_diffusion.train_utils import load_autoencoder


def collect_predictions(autoencoder, physical_model, dynamics_model, loader, state_indices, device):
    all_pred_f, all_true_f, all_images = [], [], []
    with torch.no_grad():
        for batch in loader:
            image_t = batch["image_t"].to(device)
            image_t2 = batch["image_t2"].to(device)
            image_t1 = batch["image_t1"].to(device)
            states_t2 = batch["state_t2"].to(device)
            action_t1 = batch["action_t1"].to(device)

            # Encode images to physical states
            mu_t, _ = autoencoder.encode(image_t)
            mu_t1, _ = autoencoder.encode(image_t1)
            f_t, _ = physical_model(mu_t)
            f_t1, _ = physical_model(mu_t1)

            # Predict f_t+2
            f_pred = dynamics_model(f_t, f_t1, action_t1)

            # True f_t+2
            f_true = states_t2[:, state_indices]

            all_pred_f.append(f_pred.cpu())
            all_true_f.append(f_true.cpu())
            all_images.append(image_t2.cpu())

    return (
        torch.cat(all_pred_f, dim=0).numpy(),
        torch.cat(all_true_f, dim=0).numpy(),
        torch.cat(all_images, dim=0),
    )


def plot_scatter(pred, true, state_names, output_path):
    n = len(state_names)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, name in enumerate(state_names):
        ax = axes[i]
        ax.scatter(true[:, i], pred[:, i], s=4, alpha=0.3, color="steelblue")
        lo = min(true[:, i].min(), pred[:, i].min())
        hi = max(true[:, i].max(), pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="ideal")
        rmse = float(np.sqrt(np.mean((pred[:, i] - true[:, i]) ** 2)))
        ss_res = np.sum((pred[:, i] - true[:, i]) ** 2)
        ss_tot = np.sum((true[:, i] - true[:, i].mean()) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        ax.set_title(f"{name}  (RMSE={rmse:.4f}  R²={r2:.3f})")
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
        ax.legend(fontsize=7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Dynamics model: predicted vs true state f_t+2", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"saved {output_path}")


def plot_overlay(images, pred_states, true_states, n_show, output_path):
    """Draw green (true) and red (predicted) position dots on real frames."""
    indices = np.random.choice(len(images), size=min(n_show, len(images)), replace=False)
    ncols = min(8, n_show)
    nrows = (len(indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    axes = np.array(axes).flatten()

    pred_t = torch.from_numpy(pred_states)
    true_t = torch.from_numpy(true_states)

    for plot_i, data_i in enumerate(indices):
        ax = axes[plot_i]
        img = images[data_i].permute(1, 2, 0).numpy()
        img_h, img_w = img.shape[:2]

        ax.imshow(img)
        ax.axis("off")

        # true position dot (green)
        tx, ty = state_xy_to_pixel(true_t[data_i : data_i + 1], img_h, img_w)
        ax.plot(float(tx[0]), float(ty[0]), "o", color="lime", markersize=6,
                markeredgecolor="black", markeredgewidth=0.5)

        # predicted position dot (red)
        px, py = state_xy_to_pixel(pred_t[data_i : data_i + 1], img_h, img_w)
        ax.plot(float(px[0]), float(py[0]), "o", color="red", markersize=6,
                markeredgecolor="black", markeredgewidth=0.5)

    for j in range(len(indices), len(axes)):
        axes[j].set_visible(False)

    green_patch = mpatches.Patch(color="lime", label="true position")
    red_patch = mpatches.Patch(color="red", label="predicted position")
    fig.legend(handles=[green_patch, red_patch], loc="lower center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Dynamics model position overlay (green=true, red=predicted)", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--physical_checkpoint", required=True)
    parser.add_argument("--dynamics_checkpoint", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/dynamics_eval")
    parser.add_argument("--max_test_files", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_overlay", type=int, default=32,
                        help="number of frames to show in the overlay grid")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require_visible", action="store_true",
                        help="Only eval on triplets where lander is fully on-screen")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("device:", device)

    autoencoder, ae_ckpt = load_autoencoder(args.autoencoder_checkpoint, device)
    latent_dim = int(ae_ckpt["args"]["latent_dim"])

    # load physical encoder
    phys_ckpt = torch.load(args.physical_checkpoint, map_location=device, weights_only=False)
    state_indices = phys_ckpt.get("state_indices", [0, 1, 2, 3, 4, 5])
    state_names = phys_ckpt.get("state_names", [f"s{i}" for i in state_indices])
    state_dim = len(state_indices)
    physical_model = PhysicalAutoencoder(
        latent_dim=latent_dim,
        state_dim=state_dim,
        hidden_dim=phys_ckpt["args"].get("hidden_dim", 256),
        num_layers=phys_ckpt["args"].get("num_layers", 3),
    ).to(device)
    physical_model.load_state_dict(phys_ckpt["model_state_dict"])
    physical_model.eval()

    # load dynamics
    dyn_ckpt = torch.load(args.dynamics_checkpoint, map_location=device, weights_only=False)
    dynamics_model = LunarSecondOrderDynamics(state_dim=state_dim).to(device)
    dynamics_model.load_state_dict(dyn_ckpt["model_state_dict"])
    dynamics_model.eval()

    print(f"state variables: {list(zip(state_indices, state_names))}")

    state_idx_tensor = torch.tensor(state_indices, dtype=torch.long)
    ds = LunarTripletDataset(args.test_dir, max_files=args.max_test_files,
                             require_visible=args.require_visible)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"test triplets: {len(ds)}")

    pred, true, images = collect_predictions(
        autoencoder, physical_model, dynamics_model, loader, state_idx_tensor, device
    )

    print(f"collected {len(pred)} predictions")
    for i, name in enumerate(state_names):
        rmse = float(np.sqrt(np.mean((pred[:, i] - true[:, i]) ** 2)))
        ss_res = np.sum((pred[:, i] - true[:, i]) ** 2)
        ss_tot = np.sum((true[:, i] - true[:, i].mean()) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        print(f"  {name}: RMSE={rmse:.4f}  R²={r2:.3f}  true=[{true[:,i].min():.2f}, {true[:,i].max():.2f}]  pred=[{pred[:,i].min():.2f}, {pred[:,i].max():.2f}]")

    plot_scatter(pred, true, state_names, os.path.join(args.output_dir, "scatter.png"))

    # overlay only makes sense if x and y are in the state
    has_xy = 0 in state_indices and 1 in state_indices
    if has_xy:
        xi = state_indices.index(0)
        yi = state_indices.index(1)
        pred_xy = np.stack([pred[:, xi], pred[:, yi]], axis=1)
        true_xy = np.stack([true[:, xi], true[:, yi]], axis=1)
        plot_overlay(images, pred_xy, true_xy, args.n_overlay,
                     os.path.join(args.output_dir, "overlay.png"))
    else:
        print("skipping overlay: state does not include both x (0) and y (1)")


if __name__ == "__main__":
    main()
