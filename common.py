from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def soft_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross entropy for integer labels or probability/soft labels."""
    if targets.ndim == 1:
        return F.cross_entropy(logits, targets.long())
    if targets.shape != logits.shape:
        raise ValueError(
            f"Soft targets must have shape {tuple(logits.shape)}, "
            f"received {tuple(targets.shape)}."
        )
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def one_hot(labels: Tensor, num_classes: int, dtype: Optional[torch.dtype] = None) -> Tensor:
    result = F.one_hot(labels.long(), num_classes=num_classes)
    return result.to(dtype=dtype or torch.float32)


def different_domain_permutation(domains: Tensor) -> Tensor:
    """
    Return a permutation that pairs each sample with a sample from another
    domain whenever such a sample exists.
    """
    if domains.ndim != 1:
        raise ValueError("domains must have shape [batch].")
    n = domains.numel()
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=domains.device)

    all_idx = torch.arange(n, device=domains.device)
    perm = torch.empty_like(all_idx)
    for i in range(n):
        candidates = all_idx[domains != domains[i]]
        if candidates.numel() == 0:
            candidates = all_idx[all_idx != i]
        if candidates.numel() == 0:
            perm[i] = i
        else:
            j = torch.randint(candidates.numel(), (1,), device=domains.device)
            perm[i] = candidates[j]
    return perm


def _group_norm(num_channels: int) -> nn.GroupNorm:
    groups = min(32, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def build_resnet50_featurizer(
    pretrained: bool = True,
    group_norm: bool = False,
) -> Tuple[nn.Module, int]:
    """
    Construct a torchvision ResNet-50 whose FC layer is replaced by Identity.

    For GroupNorm, ImageNet BatchNorm weights are loaded non-strictly:
    convolutional and affine parameters are reused, while BN running statistics
    are ignored.
    """
    try:
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    except ImportError:
        from torchvision.models import resnet50
        weights = "IMAGENET1K_V1" if pretrained else None

    if not group_norm:
        model = resnet50(weights=weights)
    else:
        model = resnet50(weights=None, norm_layer=_group_norm)
        if pretrained:
            source = resnet50(weights=weights)
            model.load_state_dict(source.state_dict(), strict=False)

    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feature_dim


class ResNet50Classifier(nn.Module):
    """Shared ResNet-50 feature extractor plus a linear classifier."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        group_norm: bool = False,
    ) -> None:
        super().__init__()
        self.featurizer, self.feature_dim = build_resnet50_featurizer(
            pretrained=pretrained,
            group_norm=group_norm,
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.num_classes = int(num_classes)

    def forward_features(self, x: Tensor) -> Tensor:
        return self.featurizer(x)

    def forward(self, x: Tensor, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

    def compute_loss(self, x: Tensor, y: Tensor):
        logits, features = self(x, return_features=True)
        loss = F.cross_entropy(logits, y.long())
        return {"loss": loss, "ce_loss": loss, "logits": logits, "features": features}
