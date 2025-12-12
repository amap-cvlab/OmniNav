from typing import Dict

import clip
import torch
from gym import spaces
from torch import nn as nn
from torchvision import transforms as T

from ovon.models.encoders.depth_encoder import copy_depth_encoder
from ovon.task.sensors import (
    ClipImageGoalSensor,
)


class ResNetCLIPEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        backbone_type="attnpool",
        clip_model="RN50",
        depth_ckpt: str = "",
    ):
        super().__init__()

        self.backbone_type = backbone_type
        self.rgb = "rgb" in observation_space.spaces
        self.depth = "depth" in observation_space.spaces

        if not self.is_blind:
            model, preprocess = clip.load(clip_model)

            # expected input: H x W x C (np.uint8 in [0-255])
            if (
                observation_space.spaces["rgb"].shape[0] != 224
                or observation_space.spaces["rgb"].shape[1] != 224
            ):
                print("Using CLIP preprocess for resizing+cropping to 224x224")
                preprocess_transforms = [
                    # resize and center crop to 224
                    preprocess.transforms[0],
                    preprocess.transforms[1],
                ]
            else:
                preprocess_transforms = []
            preprocess_transforms.extend(
                [
                    # already tensor, but want float
                    T.ConvertImageDtype(torch.float),
                    # normalize with CLIP mean, std
                    preprocess.transforms[4],
                ]
            )
            self.preprocess = T.Compose(preprocess_transforms)
            # expected output: H x W x C (np.float32)

            self.backbone = model.visual

            assert self.rgb

            if self.depth:
                assert depth_ckpt != ""
                self.depth_backbone = copy_depth_encoder(depth_ckpt)
                depth_size = 512
            else:
                self.depth_backbone = None
                depth_size = 0
            if "none" in backbone_type:
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048, 7, 7)
            elif self.using_both_clip_avg_attn_pool:
                # Adds an avg pooling head in parallel to final attention layer
                self.backbone.adaptive_avgpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
                )
                self.output_shape = (1024 + 2048,)  # attnpool + avgpool

                # Overwrite forward method to return both attnpool and avgpool
                # concatenated together (attnpool + avgpool).
                bound_method = forward_avg_attn_pool.__get__(
                    self.backbone, self.backbone.__class__
                )
                setattr(self.backbone, "forward", bound_method)
            elif self.using_only_clip_avgpool:
                self.backbone.attnpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
                )
                self.output_shape = (2048 + depth_size,)
            elif self.using_only_clip_attnpool:
                self.output_shape = (1024 + depth_size,)
                if ClipImageGoalSensor.cls_uuid in observation_space.spaces:
                    self.output_shape = (self.output_shape[0] + 1024,)

            for param in self.backbone.parameters():
                param.requires_grad = False
            for module in self.backbone.modules():
                if "BatchNorm" in type(module).__name__:
                    module.momentum = 0.0
            self.backbone.eval()

            if self.depth:
                for param in self.depth_backbone.parameters():
                    param.requires_grad = False
                self.depth_backbone.eval()

    @property
    def output_size(self):
        return self.output_shape[0]

    @property
    def is_blind(self):
        return self.rgb is False and self.depth is False

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        cnn_input = []
        if ClipImageGoalSensor.cls_uuid in observations:
            # Stack them into the same batch
            rgb_observations = torch.cat(
                [
                    observations["rgb"],
                    observations[ClipImageGoalSensor.cls_uuid],
                ],
                dim=0,
            )
        else:
            rgb_observations = observations["rgb"]

        rgb_observations = rgb_observations.permute(
            0, 3, 1, 2
        )  # BATCH x CHANNEL x HEIGHT X WIDTH
        rgb_observations = torch.stack(
            [self.preprocess(rgb_image) for rgb_image in rgb_observations]
        )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
        rgb_x = self.backbone(rgb_observations)

        if ClipImageGoalSensor.cls_uuid in observations:
            # Split them back out
            if self.using_both_clip_avg_attn_pool:
                rgb_x, goal_x = rgb_x
            else:
                rgb_x, goal_x = rgb_x[:, :-1024], rgb_x[:, -1024:]
            cnn_input.append(goal_x.type(torch.float32))

        cnn_input.append(rgb_x.type(torch.float32))

        if self.depth and "depth" in observations:
            depth_feats = self.depth_backbone({"depth": observations["depth"]})
            cnn_input.append(depth_feats)

        x = torch.cat(cnn_input, dim=1)

        return x

    @property
    def using_only_clip_attnpool(self):
        return "attnpool" in self.backbone_type

    @property
    def using_only_clip_avgpool(self):
        return "avgpool" in self.backbone_type

    @property
    def using_both_clip_avg_attn_pool(self):
        return "avgattnpool" in self.backbone_type


def forward_avg_attn_pool(self, x):
    """
    Adapted from https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L138
    Expects a batch of images where the batch number is even. The whole batch
    is passed through all layers except the last layer; the first half of the
    batch will be passed through avgpool and the second half will be passed
    through attnpool. The outputs of both pools are concatenated returned.
    """

    assert hasattr(self, "adaptive_avgpool")
    assert x.shape[0] % 2 == 0, "Batch size must be even"

    def stem(x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    x = x.type(self.conv1.weight.dtype)
    x = stem(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x_avgpool, x_attnpool = x.chunk(2, dim=0)
    x_avgpool = self.adaptive_avgpool(x_avgpool)
    x_attnpool = self.attnpool(x_attnpool)

    return x_avgpool, x_attnpool
