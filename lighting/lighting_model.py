"""
lighting_model.py
ResNet-18 backbone with a regression head for 3D light direction prediction.
Output is L2-normalised to unit vector.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class LightingEstimator(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Keep everything except the final classification layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # outputs (B, 512, 1, 1)

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        feat = self.features(x)          # (B, 512, 1, 1)
        out  = self.regressor(feat)      # (B, 3)
        # L2 normalise — output is always a unit vector
        out  = nn.functional.normalize(out, dim=1)
        return out
