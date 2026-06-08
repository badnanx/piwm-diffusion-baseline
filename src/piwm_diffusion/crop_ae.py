"""Small VAE for 24x24 lander crop patches (P4 compositional generation)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CropVAE(nn.Module):
    """
    Tiny VAE for 24x24 RGB lander patches.

    24x24 -> conv -> 12x12 -> conv -> 6x6 -> flatten -> mu/logvar (16-dim)
    16-dim -> fc -> 6x6 -> deconv -> 12x12 -> deconv -> 24x24
    """

    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_dim = 64 * 6 * 6  # 2304

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # (B, 32, 12, 12)
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # (B, 64, 6, 6)
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, self.feature_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # (B, 32, 12, 12)
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),   # (B, 3, 24, 24)
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x).reshape(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z).reshape(z.size(0), 64, 6, 6)
        return self.decoder_conv(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        recon = self.decode(self.reparameterize(mu, logvar))
        return recon, mu, logvar


class SpriteVAE(nn.Module):
    """
    VAE for sprite_size×sprite_size RGB lander sprites.

    Works for any sprite_size divisible by 4.
    Default 32×32: 32→16→8 (feature_dim = 64*8*8 = 4096)
    """

    def __init__(self, latent_dim: int = 16, sprite_size: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.sprite_size = sprite_size
        s = sprite_size // 4
        self.feature_dim = 64 * s * s
        self._s = s

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.fc_mu     = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.feature_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x).reshape(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z).reshape(z.size(0), 64, self._s, self._s)
        return self.decoder_conv(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def crop_vae_loss(
    recon: torch.Tensor,
    image: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 0.001,
) -> dict[str, torch.Tensor]:
    recon_loss = F.mse_loss(recon, image, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return {
        "loss": recon_loss + kl_weight * kl_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
    }
