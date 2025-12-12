import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from habitat_baselines.rl.ddppo.policy.resnet import resnet18
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from torch import nn as nn


class ResNet18DepthEncoder(nn.Module):
    def __init__(self, depth_encoder, visual_fc):
        super().__init__()
        self.encoder = depth_encoder
        self.visual_fc = visual_fc

    def forward(self, x):
        x = self.encoder(x)
        x = self.visual_fc(x)
        return x

    def load_state_dict(self, state_dict, strict: bool = True):
        # TODO: allow dicts trained with both attn and avg pool to be loaded
        ignore_attnpool = False
        if ignore_attnpool:
            pass
        return super().load_state_dict(state_dict, strict=strict)


def copy_depth_encoder(depth_ckpt):
    """
    Returns an encoder that stacks the encoder and visual_fc of the provided
    depth checkpoint
    :param depth_ckpt: path to a resnet18 depth pointnav policy
    :return: nn.Module representing the backbone of the depth policy
    """
    # Initialize encoder and fc layers
    base_planes = 32
    ngroups = 32
    spatial_size = 128

    observation_space = SpaceDict(
        {
            "depth": spaces.Box(
                low=0.0, high=1.0, shape=(256, 256, 1), dtype=np.float32
            ),
        }
    )
    depth_encoder = ResNetEncoder(
        observation_space,
        base_planes,
        ngroups,
        spatial_size,
        make_backbone=resnet18,
    )

    flat_output_shape = 2048
    hidden_size = 512
    visual_fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(flat_output_shape, hidden_size),
        nn.ReLU(True),
    )

    pretrained_state = torch.load(depth_ckpt, map_location="cpu")

    # Load weights into depth encoder
    depth_encoder_state_dict = {
        k[len("actor_critic.net.visual_encoder.") :]: v
        for k, v in pretrained_state["state_dict"].items()
        if k.startswith("actor_critic.net.visual_encoder.")
    }
    depth_encoder.load_state_dict(depth_encoder_state_dict)

    # Load weights in fc layers
    visual_fc_state_dict = {
        k[len("actor_critic.net.visual_fc.") :]: v
        for k, v in pretrained_state["state_dict"].items()
        if k.startswith("actor_critic.net.visual_fc.")
    }
    visual_fc.load_state_dict(visual_fc_state_dict)

    modified_depth_encoder = ResNet18DepthEncoder(depth_encoder, visual_fc)

    return modified_depth_encoder
