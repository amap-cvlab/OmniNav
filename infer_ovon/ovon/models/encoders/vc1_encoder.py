import torch
import torch.nn as nn
from habitat_baselines.common.tensor_dict import TensorDict

from ovon.obs_transformers.resize import image_resize


class VC1Encoder(nn.Module):
    def __init__(self):
        from vc_models.models.vit import model_utils

        super().__init__()
        (
            self.model,
            self.output_size,
            self.model_transforms,
            model_info,
        ) = model_utils.load_model(model_utils.VC1_BASE_NAME)
        self.output_shape = (self.output_size,)

    def forward(self, observations: "TensorDict", *args, **kwargs) -> torch.Tensor:
        rgb = observations["rgb"]
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0

        if rgb.shape[1] != 224 and rgb.shape[2] != 224:
            # The img loaded should be Bx3x250x250
            rgb = image_resize(rgb, size=(250, 250), channels_last=True)

        assert rgb.shape[1] == 224 and rgb.shape[2] == 224
        # Change the channels to be first
        rgb = rgb.permute(0, 3, 1, 2)
        with torch.inference_mode():
            # Output will be of size Bx3x224x224
            x = self.model_transforms(rgb)
            # Embedding will be 1x768
            x = self.model(x)

        return x
