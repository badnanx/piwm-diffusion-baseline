"""
VQ-VAE for Lunar Lander — discrete latent variant of PiwmConvVAE.
Drop-in replacement with same encoder/decoder, VQ bottleneck instead of VAE reparameterization.
"""
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQLayer(nn.Module):
    """Vector Quantization layer for flat latents."""

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, D) pre-quantization latent

        Returns:
            z_quant: (B, D) quantized latent (embeddings, continuous)
            vq_loss: scalar VQ commitment loss
            perplexity: scalar (entropy of code usage)
        """
        # Find nearest embeddings
        distances = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(z, self.embeddings.weight.t())
        )
        indices = torch.argmin(distances, dim=1)
        z_quant = self.embeddings(indices)

        # VQ loss: commitment loss pushes z toward embeddings, codebook loss pushes embeddings toward z
        e_loss = F.mse_loss(z_quant.detach(), z)
        q_loss = F.mse_loss(z_quant, z.detach())
        vq_loss = q_loss + self.beta * e_loss

        # Straight-through estimator: copy gradients from z_quant back to z
        z_quant = z + (z_quant - z).detach()

        # Perplexity: measure of codebook usage (1 = uniform, > 1 = skewed)
        avg_probs = torch.bincount(indices, minlength=self.num_embeddings).float() / len(indices)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_quant, vq_loss, perplexity


class PiwmVQVAE(nn.Module):
    """
    VQ-VAE for Lunar Lander images (100x150x3).

    Encoder: image → (256, 6, 9) features
    VQ: (256*6*9,) → 48-dim code → 48-dim embedding
    Decoder: 48-dim embedding → (256, 6, 9) features → image

    Compatible with downstream physical encoder and DDPM (expect continuous 48-dim latents from embeddings).
    """

    def __init__(
        self,
        latent_dim: int = 48,
        num_codebook_vectors: int = 512,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_codebook_vectors = num_codebook_vectors

        # Encoder: same as PiwmConvVAE
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.feature_shape = (256, 6, 9)
        self.feature_dim = 256 * 6 * 9

        # Project to latent space for VQ
        self.fc_pre_quant = nn.Linear(self.feature_dim, latent_dim)
        self.vq_layer = VQLayer(num_codebook_vectors, latent_dim, beta=beta)
        self.fc_post_quant = nn.Linear(latent_dim, self.feature_dim)

        # Decoder: same as PiwmConvVAE
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image to quantized embeddings (continuous 48-dim vectors)."""
        h = self.encoder_conv(x)
        h = h.reshape(h.size(0), -1)
        z_pre = self.fc_pre_quant(h)
        z_quant, _, _ = self.vq_layer(z_pre)
        return z_quant, torch.zeros_like(z_quant)  # Return (z_quant, dummy) for API compatibility

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent (48-dim continuous embedding) to image."""
        h = self.fc_post_quant(z)
        h = h.reshape(z.size(0), *self.feature_shape)
        recon = self.decoder_conv(h)
        return F.interpolate(recon, size=(100, 150), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, 100, 150) image

        Returns:
            recon: (B, 3, 100, 150) reconstructed image
            vq_loss: scalar VQ loss
            perplexity: scalar perplexity metric
            z_pre: (B, latent_dim) pre-quantization features (used for P1 state supervision)
        """
        h = self.encoder_conv(x)
        h = h.reshape(h.size(0), -1)
        z_pre = self.fc_pre_quant(h)

        z_quant, vq_loss, perplexity = self.vq_layer(z_pre)

        recon = self.decode(z_quant)
        return recon, vq_loss, perplexity, z_pre


def vq_vae_loss(
    recon: torch.Tensor,
    image: torch.Tensor,
    vq_loss: torch.Tensor,
    state: torch.Tensor,
    state_indices: Sequence[int],
    z_pre: torch.Tensor,
    state_weight: float = 0.0,
    recon_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    VQ-VAE loss for PIWM.

    Args:
        recon: reconstructed image
        image: target image
        vq_loss: VQ commitment loss from layer
        state: (B, 8) full state vector
        state_indices: which state dims to supervise
        z_pre: (B, latent_dim) pre-quantization features for P1 supervision
        state_weight: weight on P1 state loss (MSE z_pre[:, :k] vs true state)
        recon_weight: weight on reconstruction loss
    """
    recon_loss = F.mse_loss(recon, image, reduction="mean")
    total = recon_weight * recon_loss + vq_loss

    state_loss = torch.zeros((), device=recon.device)
    if state_weight > 0.0:
        k = len(state_indices)
        target = state[:, state_indices]
        state_loss = F.mse_loss(z_pre[:, :k], target, reduction="mean")
        total = total + state_weight * state_loss

    return {
        "loss": total,
        "recon_loss": recon_loss,
        "vq_loss": vq_loss,
        "state_loss": state_loss,
    }
