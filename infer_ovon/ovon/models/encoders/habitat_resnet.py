from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder


class HabitatResNetEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.backbone = ResNetEncoder(**kwargs)
        self.output_size = np.prod(self.backbone.output_shape)
        self.output_shape = self.backbone.output_shape

    def forward(self, observations: Dict, *args, **kwargs) -> torch.Tensor:
        return self.backbone(observations)
