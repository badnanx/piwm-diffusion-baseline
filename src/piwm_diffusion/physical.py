import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalEncoderMLP(nn.Module):
    """Map visual autoencoder latents to an interpretable physical state."""

    def __init__(
        self,
        latent_dim: int = 64,
        state_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatentDecoderMLP(nn.Module):
    """Map an interpretable physical state back into visual latent space."""

    def __init__(
        self,
        state_dim: int = 6,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class PhysicalAutoencoder(nn.Module):
    """
    Extrinsic PIWM bridge:

        visual latent z -> physical state f -> reconstructed visual latent z_hat
    """

    def __init__(
        self,
        latent_dim: int = 64,
        state_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = PhysicalEncoderMLP(latent_dim, state_dim, hidden_dim, num_layers)
        self.decoder = LatentDecoderMLP(state_dim, latent_dim, hidden_dim, num_layers)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        physical = self.encoder(z)
        z_recon = self.decoder(physical)
        return physical, z_recon


def physical_autoencoder_loss(
    pred_state: torch.Tensor,
    target_state: torch.Tensor,
    z_recon: torch.Tensor,
    z: torch.Tensor,
    state_weight: float = 1.0,
    latent_recon_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    state_loss = F.mse_loss(pred_state, target_state, reduction="mean")
    latent_recon_loss = F.mse_loss(z_recon, z, reduction="mean")
    loss = state_weight * state_loss + latent_recon_weight * latent_recon_loss
    return {
        "loss": loss,
        "state_loss": state_loss,
        "latent_recon_loss": latent_recon_loss,
    }
