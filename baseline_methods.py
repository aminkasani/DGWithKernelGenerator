from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import (
    build_resnet50_featurizer,
    different_domain_permutation,
    one_hot,
    soft_cross_entropy,
)


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, coefficient: float):
        ctx.coefficient = float(coefficient)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.coefficient * grad_output, None


def grad_reverse(x: Tensor, coefficient: float = 1.0) -> Tensor:
    return _GradientReversal.apply(x, coefficient)


class DMModel(nn.Module):
    """
    Practical source-only DG reconstruction of Domain Mixup (DM).

    It performs:
      1) pixel-level cross-domain mixup,
      2) feature-level cross-domain mixup,
      3) adversarial source-domain prediction through gradient reversal.

    The original DM-ADA paper uses a larger VAE/GAN adaptation framework.
    Since DaNet evaluates a source-only DG setting and does not document that
    conversion, this module is an explicit multi-source DG adaptation rather
    than a claim of bit-exact reproduction.
    """

    def __init__(
        self,
        num_classes: int,
        num_domains: int,
        pretrained: bool = True,
        alpha: float = 0.2,
        pixel_weight: float = 1.0,
        feature_weight: float = 1.0,
        domain_weight: float = 0.1,
        grl_coefficient: float = 1.0,
        discriminator_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_domains = int(num_domains)
        self.alpha = float(alpha)
        self.pixel_weight = float(pixel_weight)
        self.feature_weight = float(feature_weight)
        self.domain_weight = float(domain_weight)
        self.grl_coefficient = float(grl_coefficient)

        self.featurizer, self.feature_dim = build_resnet50_featurizer(
            pretrained=pretrained
        )
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(self.feature_dim, discriminator_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(discriminator_hidden, self.num_domains),
        )

    def forward_features(self, x: Tensor) -> Tensor:
        return self.featurizer(x)

    def forward(self, x: Tensor, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits

    def _sample_lambda(self, n: int, device, dtype) -> Tensor:
        beta = torch.distributions.Beta(self.alpha, self.alpha)
        return beta.sample((n,)).to(device=device, dtype=dtype)

    def compute_loss(self, x: Tensor, y: Tensor, domains: Tensor):
        n = x.shape[0]
        perm = different_domain_permutation(domains)
        lam = self._sample_lambda(n, x.device, x.dtype)

        class_targets = one_hot(y, self.num_classes, dtype=x.dtype)
        domain_targets = one_hot(domains, self.num_domains, dtype=x.dtype)
        mixed_class_targets = (
            lam[:, None] * class_targets
            + (1.0 - lam[:, None]) * class_targets[perm]
        )
        mixed_domain_targets = (
            lam[:, None] * domain_targets
            + (1.0 - lam[:, None]) * domain_targets[perm]
        )

        features = self.forward_features(x)
        logits = self.classifier(features)
        ce = F.cross_entropy(logits, y.long())

        lam_x = lam.view(n, 1, 1, 1)
        mixed_x = lam_x * x + (1.0 - lam_x) * x[perm]
        pixel_features = self.forward_features(mixed_x)
        pixel_logits = self.classifier(pixel_features)
        pixel_cls = soft_cross_entropy(pixel_logits, mixed_class_targets)

        mixed_features = (
            lam[:, None] * features
            + (1.0 - lam[:, None]) * features[perm]
        )
        feature_logits = self.classifier(mixed_features)
        feature_cls = soft_cross_entropy(feature_logits, mixed_class_targets)

        domain_logits = self.domain_discriminator(
            grad_reverse(features, self.grl_coefficient)
        )
        pixel_domain_logits = self.domain_discriminator(
            grad_reverse(pixel_features, self.grl_coefficient)
        )
        feature_domain_logits = self.domain_discriminator(
            grad_reverse(mixed_features, self.grl_coefficient)
        )

        domain_loss = (
            F.cross_entropy(domain_logits, domains.long())
            + soft_cross_entropy(pixel_domain_logits, mixed_domain_targets)
            + soft_cross_entropy(feature_domain_logits, mixed_domain_targets)
        ) / 3.0

        loss = (
            ce
            + self.pixel_weight * pixel_cls
            + self.feature_weight * feature_cls
            + self.domain_weight * domain_loss
        )
        return {
            "loss": loss,
            "ce_loss": ce,
            "pixel_mix_loss": pixel_cls,
            "feature_mix_loss": feature_cls,
            "domain_loss": domain_loss,
            "logits": logits,
            "features": features,
            "lambda": lam,
            "permutation": perm,
        }
