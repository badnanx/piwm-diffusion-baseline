import json
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_autoencoder(checkpoint_path: str, device: torch.device) -> tuple[nn.Module, dict]:
    """Load either a PiwmConvVAE or PiwmVQVAE from a checkpoint, dispatching on model_type."""
    from piwm_diffusion.autoencoder import PiwmConvVAE
    from piwm_diffusion.vq_autoencoder import PiwmVQVAE

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    latent_dim = int(ckpt["args"]["latent_dim"])
    model_type = ckpt.get("model_type", "continuous")

    if model_type == "vq_vae":
        model = PiwmVQVAE(
            latent_dim=latent_dim,
            num_codebook_vectors=int(ckpt["args"].get("num_codebook_vectors", 512)),
            beta=float(ckpt["args"].get("beta", 0.25)),
        ).to(device)
    else:
        model = PiwmConvVAE(latent_dim=latent_dim).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, ckpt


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }
