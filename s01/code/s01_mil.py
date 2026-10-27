"""Patient-level mean/attention MIL heads for cached recording embeddings."""

from __future__ import annotations

import math
from numbers import Integral

import torch
from torch import nn


class PatientMIL(nn.Module):
    """Project [B,R,H] recording embeddings and pool only mask=True recordings."""

    def __init__(self, hidden_size: int, pooling: str = "mean", projection_size: int = 256,
                 dropout: float = 0.3, attention_size: int = 128):
        super().__init__()
        if any(isinstance(size, bool) or not isinstance(size, int) or size < 1
               for size in (hidden_size, projection_size, attention_size)):
            raise ValueError("Hidden, projection, and attention sizes must be positive integers.")
        if pooling not in {"mean", "attention"}:
            raise ValueError("pooling must be mean or attention.")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and between zero and one (exclusive).")
        self.hidden_size = hidden_size
        self.pooling = pooling
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, projection_size),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.attention = (nn.Sequential(nn.Linear(projection_size, attention_size),
                                       nn.Tanh(), nn.Linear(attention_size, 1, bias=False))
                          if pooling == "attention" else None)
        self.classifier = nn.Linear(projection_size, 3)

    def forward(self, embeddings, mask, return_attention: bool = False):
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.hidden_size:
            raise ValueError("embeddings must have shape [B, R, hidden_size].")
        if not embeddings.is_floating_point():
            raise ValueError("embeddings must contain floating-point values.")
        if mask.dtype != torch.bool or mask.shape != embeddings.shape[:2]:
            raise ValueError("mask must be bool [B, R], with True for valid recordings.")
        if mask.device != embeddings.device:
            raise ValueError("embeddings and mask must be on the same device.")
        if embeddings.shape[0] == 0 or not mask.any(dim=1).all():
            raise ValueError("Every patient bag needs at least one valid recording.")
        if not torch.isfinite(embeddings[mask]).all():
            raise ValueError("Valid recording embeddings must be finite.")
        # Replace padding before projection, so even NaN padding cannot contaminate a bag.
        projected = self.projection(embeddings.masked_fill(~mask.unsqueeze(-1), 0))
        if self.attention is None:
            weights = mask.to(projected.dtype) / mask.sum(dim=1, keepdim=True)
        else:
            scores = self.attention(projected).squeeze(-1).masked_fill(~mask, -torch.inf)
            weights = scores.softmax(dim=1)
        pooled = (projected * weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(pooled)
        return (logits, weights) if return_attention else logits


def patient_class_weights(targets) -> torch.Tensor:
    """Inverse-frequency weights n/(3*class_count) from training patients only."""

    targets = torch.as_tensor(targets)
    if (targets.ndim != 1 or targets.numel() == 0 or targets.dtype == torch.bool
            or targets.is_floating_point() or targets.is_complex()):
        raise ValueError("targets must be a nonempty one-dimensional integer sequence.")
    if (targets < 0).any() or (targets > 2).any():
        raise ValueError("targets must use class indices zero, one, and two.")
    counts = torch.bincount(targets.long(), minlength=3).float()
    if (counts == 0).any():
        raise ValueError("Training patients must include all three classes.")
    return targets.numel() / (3 * counts)


def collate_bags(batch: list[dict]) -> dict:
    """Pad variable recording counts; return embeddings, mask, targets, patient_ids."""

    if not batch:
        raise ValueError("Cannot collate an empty patient batch.")
    for bag in batch:
        if not isinstance(bag, dict) or not {"embeddings", "target", "patient_id"}.issubset(bag):
            raise ValueError("Each bag requires embeddings, target, and patient_id.")
        values = bag["embeddings"]
        if (not isinstance(values, torch.Tensor) or values.ndim != 2
                or min(values.shape) < 1 or not values.is_floating_point()
                or not torch.isfinite(values).all()):
            raise ValueError("Each bag needs finite floating embeddings shaped [R, H], with R,H > 0.")
        target = bag["target"]
        if isinstance(target, bool) or not isinstance(target, Integral) or not 0 <= target < 3:
            raise ValueError("Every bag target must be an integer class index zero, one, or two.")
        if not isinstance(bag["patient_id"], str) or not bag["patient_id"].strip():
            raise ValueError("Every bag needs a nonblank patient_id.")
    first = batch[0]["embeddings"]
    if any(bag["embeddings"].shape[1] != first.shape[1]
           or bag["embeddings"].dtype != first.dtype
           or bag["embeddings"].device != first.device for bag in batch):
        raise ValueError("All bags must share embedding width, dtype, and device.")
    maximum = max(len(bag["embeddings"]) for bag in batch)
    padded = first.new_zeros((len(batch), maximum, first.shape[1]))
    mask = torch.zeros((len(batch), maximum), dtype=torch.bool, device=first.device)
    for index, bag in enumerate(batch):
        count = len(bag["embeddings"])
        padded[index, :count] = bag["embeddings"]
        mask[index, :count] = True
    return {"embeddings": padded, "mask": mask,
            "targets": torch.tensor([bag["target"] for bag in batch], dtype=torch.long,
                                    device=first.device),
            "patient_ids": [bag["patient_id"] for bag in batch]}
