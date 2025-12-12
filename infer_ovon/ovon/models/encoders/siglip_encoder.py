import timm
import torch
import torch.nn as nn
from habitat_baselines.common.tensor_dict import TensorDict
from torchvision import transforms


class SigLIPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            model_name="vit_base_patch16_siglip_256",
            pretrained=True,
            num_classes=0,
        )
        self.model = self.model.eval()
        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    size=(256, 256), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.Normalize(
                    mean=torch.tensor([0.5000, 0.5000, 0.5000]),
                    std=torch.tensor([0.5000, 0.5000, 0.5000]),
                ),
            ]
        )
        self.output_size = 768
        self.output_shape = (self.output_size,)

    def forward(self, observations: TensorDict, *args, **kwargs) -> torch.Tensor:
        rgb = observations["rgb"]
        rgb = rgb.permute(0, 3, 1, 2)  # NHWC -> NCHW
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        else:
            assert (rgb >= 0.0).all() and (rgb <= 1.0).all()
        with torch.inference_mode():
            # Output will be of size Bx3x224x224
            x = self.transforms(rgb)
            # Embedding will be 1x768
            x = self.model(x)

        return x
