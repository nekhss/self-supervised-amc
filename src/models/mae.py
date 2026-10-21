
"""Masked Autoencoder for raw-IQ automatic modulation classification."""

from typing import Optional

import torch
import torch.nn as nn


class IQMaskedAutoencoder(nn.Module):
    """Reconstruct masked temporal patches from raw complex IQ samples.

    Input shape:  [B, 2, 1024]
    Output prediction shape: [B, 16, 128]

    Each patch contains 64 time steps from both IQ channels,
    flattened to 128 real-valued features.
    """

    def __init__(
        self,
        signal_length: int = 1024,
        patch_size: int = 64,
        embed_dim: int = 192,
        encoder_depth: int = 4,
        encoder_heads: int = 4,
        decoder_dim: int = 128,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        mask_ratio: float = 0.5,
    ):
        super().__init__()

        if signal_length <= 0 or patch_size <= 0:
            raise ValueError("signal_length and patch_size must be positive.")
        if signal_length % patch_size != 0:
            raise ValueError("signal_length must be divisible by patch_size.")
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1.")
        if embed_dim % encoder_heads != 0:
            raise ValueError("embed_dim must be divisible by encoder_heads.")
        if decoder_dim % decoder_heads != 0:
            raise ValueError("decoder_dim must be divisible by decoder_heads.")

        self.signal_length = signal_length
        self.patch_size = patch_size
        self.num_patches = signal_length // patch_size
        self.patch_dim = 2 * patch_size
        self.mask_ratio = mask_ratio

        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.encoder_pos = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=encoder_heads,
            dim_feedforward=4 * embed_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_depth
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_dim)
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=4 * decoder_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=decoder_depth
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_dim)

        nn.init.trunc_normal_(self.encoder_pos, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def patchify(self, signals: torch.Tensor) -> torch.Tensor:
        """Convert [B, 2, L] into [B, number_of_patches, patch_dim]."""
        if signals.ndim != 3:
            raise ValueError("Expected input shape [B, 2, signal_length].")
        if signals.shape[1] != 2:
            raise ValueError("Expected exactly two IQ channels.")
        if signals.shape[2] != self.signal_length:
            raise ValueError(
                f"Expected signal length {self.signal_length}, "
                f"received {signals.shape[2]}."
            )

        batch_size = signals.shape[0]
        patches = signals.reshape(
            batch_size, 2, self.num_patches, self.patch_size
        )
        patches = patches.permute(0, 2, 1, 3)
        return patches.reshape(batch_size, self.num_patches, self.patch_dim)

    def random_mask(
        self,
        tokens: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ):
        """Shuffle patches and retain a random visible subset."""
        batch_size, num_patches, _ = tokens.shape
        keep_count = max(
            1, min(num_patches - 1, int(num_patches * (1 - self.mask_ratio)))
        )

        noise = torch.rand(
            batch_size,
            num_patches,
            device=tokens.device,
            generator=generator,
        )
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep_count]

        visible = torch.gather(
            tokens,
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )

        # 0 = visible, 1 = masked, in original patch order.
        mask = torch.ones(
            batch_size, num_patches, device=tokens.device
        )
        mask[:, :keep_count] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return visible, mask, ids_restore

    def forward(
        self,
        signals: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ):
        """Return reconstruction, mask, and masked-patch MSE loss."""
        patches = self.patchify(signals)

        tokens = self.patch_embed(patches) + self.encoder_pos
        visible, mask, ids_restore = self.random_mask(tokens, generator)

        encoded = self.encoder_norm(self.encoder(visible))

        decoded_visible = self.decoder_embed(encoded)
        missing_count = self.num_patches - decoded_visible.shape[1]

        mask_tokens = self.mask_token.expand(
            signals.shape[0], missing_count, -1
        )
        shuffled_tokens = torch.cat(
            [decoded_visible, mask_tokens], dim=1
        )

        # Restore the original temporal patch order.
        decoded = torch.gather(
            shuffled_tokens,
            1,
            ids_restore.unsqueeze(-1).expand(
                -1, -1, shuffled_tokens.shape[-1]
            ),
        )
        decoded = decoded + self.decoder_pos
        decoded = self.decoder_norm(self.decoder(decoded))
        prediction = self.decoder_pred(decoded)

        # MAE objective: score masked patches, not visible patches.
        per_patch_loss = (prediction - patches).pow(2).mean(dim=-1)
        loss = (per_patch_loss * mask).sum() / mask.sum().clamp_min(1)

        return {
            "loss": loss,
            "prediction": prediction,
            "mask": mask,
        }
