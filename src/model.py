from __future__ import annotations

import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_model(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True):
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model
