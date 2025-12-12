import torch
import torch.hub
import torch.nn as nn
from torchvision import transforms


class DINOV2Encoder(nn.Module):
    def __init__(self, backbone_name="dinov2_vitb14", output_size=768):
        super().__init__()
        self.model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2", model=backbone_name
        )
        self.model.eval()
        self.model_transforms = self.make_depth_transform()
        self.output_size = output_size
        self.output_shape = (self.output_size,)

    def make_depth_transform(self):
        return transforms.Compose(
            [
                transforms.Normalize(
                    mean=(123.675, 116.28, 103.53),
                    std=(58.395, 57.12, 57.375),
                ),
            ]
        )

    def forward(self, observations: "TensorDict", *args, **kwargs) -> torch.Tensor:
        # rgb is a tensor of shape (batch_size, height, width, channels)
        rgb = observations["rgb"]

        # Assert that the rgb images are of type uint8
        assert rgb.dtype == torch.uint8

        # Assert that the height and width are both 224
        assert rgb.shape[1] == 224 and rgb.shape[2] == 224

        rgb = rgb.float()

        # PyTorch models expect the input in (batch, channels, height, width) format
        rgb = rgb.permute(0, 3, 1, 2)

        with torch.inference_mode():
            x = self.model(self.model_transforms(rgb))

        return x
