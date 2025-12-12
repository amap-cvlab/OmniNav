from typing import Any

from gym import spaces

from ovon.models.encoders.clip_encoder import ResNetCLIPEncoder
from ovon.models.encoders.dinov2_encoder import DINOV2Encoder
from ovon.models.encoders.siglip_encoder import SigLIPEncoder
from ovon.models.encoders.vc1_encoder import VC1Encoder

POSSIBLE_ENCODERS = [
    "clip_attnpool",
    "clip_avgpool",
    "clip_avgattnpool",
    "vc1",
    "dinov2",
    "resnet",
    "siglip",
]


def make_encoder(backbone: str, observation_space: spaces.Dict) -> Any:
    if backbone == "resnet50_clip_avgpool":
        backbone = "clip_avgpool"
        print("WARNING: resnet50_clip_avgpool is deprecated. Use clip_avgpool instead.")

    assert (
        backbone in POSSIBLE_ENCODERS
    ), f"Backbone {backbone} not found. Possible encoders: {POSSIBLE_ENCODERS}"

    if "clip" in backbone:
        backbone_type = backbone.split("_")[1]
        return ResNetCLIPEncoder(
            observation_space,
            backbone_type=backbone_type,
            clip_model="RN50",
        )
    elif backbone == "vc1":
        return VC1Encoder()
    elif backbone == "dinov2":
        return DINOV2Encoder()
    elif backbone == "siglip":
        return SigLIPEncoder()
    elif backbone == "resnet":
        resnet_baseplanes = 32
        from habitat_baselines.rl.ddppo.policy import resnet
        from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder

        return ResNetEncoder(
            observation_space=observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=resnet.resnet50,
        )
