"""GraspNet: ResNet-50 backbone with a 5-dim regression head."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class GraspNet(nn.Module):
    """ResNet-50 with the final 1000-way classifier replaced by a 5-dim regressor.

    Output is the (normalized) (x, y, theta, w, h) grasp parameter vector.
    See `src.utils.normalize_params` / `denormalize_params` for the scaling.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        self.backbone.fc = nn.Linear(2048, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


if __name__ == "__main__":
    # Build with random init (no internet needed for sanity)
    model = GraspNet(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"[model] in {tuple(x.shape)} -> out {tuple(y.shape)}")
    assert y.shape == (2, 5), y.shape
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params: {n_params:,}")
    print("[model] sanity OK")
