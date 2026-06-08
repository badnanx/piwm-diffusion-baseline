import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class ConditionalDenoiserMLP(nn.Module):
    """Small DDPM denoiser for vector latents conditioned on physical state."""

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        time_dim: int = 64,
        hidden_dim: int = 512,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        in_dim = latent_dim + cond_dim + time_dim

        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(timesteps)
        return self.net(torch.cat([x_t, t_emb, cond], dim=1))


class DiffusionSchedule:
    def __init__(
        self,
        num_steps: int,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: str | torch.device) -> "DiffusionSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self


def ddpm_noise_prediction_loss(
    model: nn.Module,
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    batch = x0.size(0)
    t = torch.randint(0, schedule.num_steps, (batch,), device=x0.device)
    noise = torch.randn_like(x0)
    alpha_bar_t = schedule.alpha_bars[t].view(batch, 1)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
    pred_noise = model(x_t, t, cond)
    return F.mse_loss(pred_noise, noise, reduction="mean")


@torch.no_grad()
def sample_latents(
    model: nn.Module,
    schedule: DiffusionSchedule,
    cond: torch.Tensor,
    latent_dim: int,
    x0_init: torch.Tensor | None = None,
    t_start: int | None = None,
    constraint_fn: "Callable[[torch.Tensor], torch.Tensor] | None" = None,
) -> torch.Tensor:
    """Sample latents from the diffusion model.

    x0_init + t_start: SDEdit — forward-diffuse x0_init to t_start, then
    reverse from there instead of from pure noise.

    constraint_fn: optional callable (z_norm) -> z_norm applied after each
    denoising step to steer latents toward physical consistency. Should be
    a closed-over function that has access to f and normalization stats.
    """
    model.eval()

    if x0_init is not None and t_start is not None and t_start > 0:
        t_start = min(t_start, schedule.num_steps - 1)
        alpha_bar = schedule.alpha_bars[t_start]
        noise = torch.randn_like(x0_init)
        x = torch.sqrt(alpha_bar) * x0_init + torch.sqrt(1.0 - alpha_bar) * noise
        end_step = t_start
    else:
        x = torch.randn(cond.size(0), latent_dim, device=cond.device)
        end_step = schedule.num_steps - 1

    for step in reversed(range(end_step + 1)):
        t = torch.full((cond.size(0),), step, device=cond.device, dtype=torch.long)
        beta_t = schedule.betas[step]
        alpha_t = schedule.alphas[step]
        alpha_bar_t = schedule.alpha_bars[step]

        pred_noise = model(x, t, cond)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
        )
        if step > 0:
            x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
        else:
            x = mean

        if constraint_fn is not None:
            x = constraint_fn(x)

    return x
