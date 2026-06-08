import os
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from piwm_diffusion.crop import state_xy_to_pixel


def save_image_grid(
    images: torch.Tensor,
    path: str,
    titles: Optional[list[str]] = None,
    nrow: int = 8,
    suptitle: Optional[str] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    images = images.detach().cpu().clamp(0.0, 1.0)
    n = images.size(0)
    nrow = min(nrow, n)
    ncol = (n + nrow - 1) // nrow

    fig, axes = plt.subplots(ncol, nrow, figsize=(2 * nrow, 2 * ncol))
    if ncol == 1:
        axes = [axes]

    for idx in range(ncol * nrow):
        ax = axes[idx // nrow][idx % nrow]
        ax.axis("off")
        if idx < n:
            ax.imshow(images[idx].permute(1, 2, 0))
            if titles and idx < len(titles):
                ax.set_title(titles[idx], fontsize=8)

    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_reconstruction_grid(
    real: torch.Tensor,
    recon: torch.Tensor,
    path: str,
    num_images: int = 8,
    suptitle: Optional[str] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    real = real[:num_images].detach().cpu().clamp(0.0, 1.0)
    recon = recon[:num_images].detach().cpu().clamp(0.0, 1.0)
    n = real.size(0)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(real[i].permute(1, 2, 0))
        axes[0, i].axis("off")
        axes[0, i].set_title("real", fontsize=8)
        axes[1, i].imshow(recon[i].permute(1, 2, 0))
        axes[1, i].axis("off")
        axes[1, i].set_title("recon", fontsize=8)

    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_three_row_comparison(
    real: torch.Tensor,
    deterministic: torch.Tensor,
    generated: torch.Tensor,
    path: str,
    num_images: int = 8,
    row_labels: tuple[str, str, str] = (
        "Real (ground truth)",
        "Deterministic (physics decoder, no diffusion)",
        "Generated (SDEdit + constraint)",
    ),
    suptitle: Optional[str] = None,
) -> None:
    """Three-row comparison: real vs deterministic vs diffusion-generated."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = min(num_images, real.size(0))
    real = real[:n].detach().cpu().clamp(0.0, 1.0)
    deterministic = deterministic[:n].detach().cpu().clamp(0.0, 1.0)
    generated = generated[:n].detach().cpu().clamp(0.0, 1.0)

    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    rows = [real, deterministic, generated]

    for row_idx, (imgs, label) in enumerate(zip(rows, row_labels)):
        for col_idx in range(n):
            ax = axes[row_idx, col_idx]
            ax.imshow(imgs[col_idx].permute(1, 2, 0))
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, labelpad=120, va="center")

    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_crop_box(ax, state: torch.Tensor, img_h: int, img_w: int, crop_size: int) -> None:
    state_batch = state.detach().cpu().view(1, -1)
    px, py = state_xy_to_pixel(state_batch, img_h, img_w)
    cx = float(px[0])
    cy = float(py[0])
    half = crop_size / 2.0
    rect = Rectangle(
        (cx - half, cy - half),
        crop_size,
        crop_size,
        fill=False,
        edgecolor="cyan",
        linewidth=2.5,
    )
    ax.add_patch(rect)


def save_reconstruction_grid_boxed(
    real: torch.Tensor,
    recon: torch.Tensor,
    states: torch.Tensor,
    path: str,
    crop_size: int = 24,
    num_images: int = 8,
    suptitle: Optional[str] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    real = real[:num_images].detach().cpu().clamp(0.0, 1.0)
    recon = recon[:num_images].detach().cpu().clamp(0.0, 1.0)
    states = states[:num_images].detach().cpu()
    n = real.size(0)
    img_h = real.shape[2]
    img_w = real.shape[3]

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(real[i].permute(1, 2, 0))
        axes[0, i].axis("off")
        axes[0, i].set_title("real", fontsize=8)
        _draw_crop_box(axes[0, i], states[i], img_h, img_w, crop_size)

        axes[1, i].imshow(recon[i].permute(1, 2, 0))
        axes[1, i].axis("off")
        axes[1, i].set_title("recon", fontsize=8)
        _draw_crop_box(axes[1, i], states[i], img_h, img_w, crop_size)

    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
