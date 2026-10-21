"""CNN-based masked autoencoder for raw-IQ signals."""

import torch
import torch.nn as nn

from src.models.cnn import IQCNNEncoder


class IQCNNMaskedAutoencoder(nn.Module):
    """Reconstruct masked IQ patches from visible patches."""

    def __init__(
        self,
        signal_length: int = 1024,
        patch_size: int = 64,
        mask_ratio: float = 0.5,
        feature_dim: int = 128,
        decoder_dim: int = 128,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
    ):
        super().__init__()

        if signal_length % patch_size != 0:
            raise ValueError("signal_length must be divisible by patch_size.")
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be strictly between 0 and 1.")
        if decoder_dim % decoder_heads != 0:
            raise ValueError("decoder_dim must be divisible by decoder_heads.")

        self.signal_length = signal_length
        self.patch_size = patch_size
        self.num_patches = signal_length // patch_size
        self.patch_dim = 2 * patch_size
        self.mask_ratio = mask_ratio
        self.feature_dim = feature_dim
        self.decoder_dim = decoder_dim

        # Shared backbone: the same patch encoder used by the classifier.
        self.encoder = IQCNNEncoder(feature_dim=feature_dim)

        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, feature_dim)
        )
        self.encoder_to_decoder = nn.Linear(feature_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_dim)
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=decoder_depth,
            norm=nn.LayerNorm(decoder_dim),
        )
        self.reconstruction_head = nn.Linear(decoder_dim, self.patch_dim)

        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def patchify(self, signals: torch.Tensor) -> torch.Tensor:
        """[B, 2, L] -> [B, N, 2 * patch_size]."""
        if signals.ndim != 3 or signals.shape[1:] != (
            2, self.signal_length
        ):
            raise ValueError(
                f"Expected [B, 2, {self.signal_length}], "
                f"got {tuple(signals.shape)}"
            )

        batch_size = signals.shape[0]
        patches = signals.reshape(
            batch_size, 2, self.num_patches, self.patch_size
        ).permute(0, 2, 1, 3)

        return patches.reshape(batch_size, self.num_patches, self.patch_dim)

    def random_mask(self, batch_size, device):
        """Return shuffled visible indices, restore indices and mask."""
        keep_count = max(
            1, int(self.num_patches * (1.0 - self.mask_ratio))
        )

        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :keep_count]

        mask_shuffled = torch.ones(
            batch_size, self.num_patches, device=device
        )
        mask_shuffled[:, :keep_count] = 0
        mask = torch.gather(mask_shuffled, 1, ids_restore)

        return ids_keep, ids_restore, mask

    def forward(self, signals: torch.Tensor):
        targets = self.patchify(signals)
        batch_size = signals.shape[0]

        ids_keep, ids_restore, mask = self.random_mask(
            batch_size, signals.device
        )
        keep_count = ids_keep.shape[1]

        # Gather visible raw patches only. Masked IQ values are not encoded.
        visible_targets = torch.gather(
            targets,
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, self.patch_dim),
        )
        visible_patches = visible_targets.reshape(
            batch_size, keep_count, 2, self.patch_size
        ).reshape(batch_size * keep_count, 2, self.patch_size)

        visible_features = self.encoder(visible_patches)
        visible_features = visible_features.reshape(
            batch_size, keep_count, self.feature_dim
        )

        visible_positions = self.encoder_pos_embed.expand(
            batch_size, -1, -1
        ).gather(
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, self.feature_dim),
        )
        visible_features = self.encoder_to_decoder(
            visible_features + visible_positions
        )

        # Tokens are currently in shuffled order: visible first, masked last.
        mask_tokens = self.mask_token.expand(
            batch_size, self.num_patches - keep_count, -1
        )
        shuffled_tokens = torch.cat(
            [visible_features, mask_tokens], dim=1
        )

        # Restore the original temporal order before decoding.
        decoder_tokens = torch.gather(
            shuffled_tokens,
            1,
            ids_restore.unsqueeze(-1).expand(
                -1, -1, self.decoder_dim
            ),
        )
        decoder_tokens = (
            decoder_tokens
            + self.decoder_pos_embed.expand(batch_size, -1, -1)
        )

        decoded = self.decoder(decoder_tokens)
        predictions = self.reconstruction_head(decoded)

        # Compute MSE on masked patches only.
        per_patch_mse = (predictions - targets).pow(2).mean(dim=-1)
        loss = (per_patch_mse * mask).sum() / mask.sum().clamp_min(1)

        return {
            "loss": loss,
            "predictions": predictions,
            "targets": targets,
            "mask": mask,
        }
