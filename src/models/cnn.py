"""Shared patch-based 1D-CNN encoder for raw-IQ modulation classification."""

import torch
import torch.nn as nn


class IQCNNEncoder(nn.Module):
    """Encode IQ patches shaped [batch, 2, patch_length]."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(128, feature_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3 or patches.shape[1] != 2:
            raise ValueError(
                f"Expected patches shaped [B, 2, patch_length], "
                f"got {tuple(patches.shape)}"
            )
        if patches.shape[-1] < 16:
            raise ValueError("Patch length must be at least 16 samples.")

        features = self.features(patches).squeeze(-1)
        return self.projection(features)


class IQCNNClassifier(nn.Module):
    """Classify full IQ signals using the shared patch encoder."""

    def __init__(
        self,
        num_classes: int = 24,
        feature_dim: int = 128,
        patch_size: int = 64,
    ):
        super().__init__()

        if patch_size < 16 or 1024 % patch_size != 0:
            raise ValueError("patch_size must be >= 16 and divide 1024.")

        self.patch_size = patch_size
        self.encoder = IQCNNEncoder(feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def encode_signal(self, signals: torch.Tensor) -> torch.Tensor:
        if signals.ndim != 3 or signals.shape[1:] != (2, 1024):
            raise ValueError(
                f"Expected signals shaped [B, 2, 1024], "
                f"got {tuple(signals.shape)}"
            )

        batch_size = signals.shape[0]
        num_patches = 1024 // self.patch_size

        patches = signals.reshape(
            batch_size, 2, num_patches, self.patch_size
        ).permute(0, 2, 1, 3)

        patches = patches.reshape(
            batch_size * num_patches, 2, self.patch_size
        )

        patch_features = self.encoder(patches)
        patch_features = patch_features.reshape(
            batch_size, num_patches, self.encoder.feature_dim
        )

        return patch_features.mean(dim=1)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode_signal(signals))
