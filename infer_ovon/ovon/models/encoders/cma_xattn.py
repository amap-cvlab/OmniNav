import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossModalAttention(nn.Module):
    def __init__(
        self, text_embedding_dim: int, rgb_embedding_dim: int, hidden_size: int = 512
    ) -> None:
        super().__init__()

        # Linear transformation for the text query and RGB key-value pairs
        self.text_q = nn.Linear(text_embedding_dim, hidden_size // 2)
        self.rgb_kv = nn.Conv1d(rgb_embedding_dim, hidden_size, 1)

        # Scale for attention
        self.register_buffer("_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5)))

        self._hidden_size = hidden_size

    @property
    def output_size(self) -> int:
        return self._hidden_size // 2

    def _attn(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        logits = torch.einsum("nc, nci -> ni", q, k)
        attn = F.softmax(logits * self._scale, dim=1)
        return torch.einsum("ni, nci -> nc", attn, v)

    def forward(self, text_embedding: Tensor, rgb_embedding: Tensor) -> Tensor:
        """
        Args:
            text_embedding: [batch_size, text_embedding_dim] tensor (language)
            rgb_embedding: [batch_size, rgb_embedding_dim] tensor (visual)

        Returns:
            [batch_size, embed_dim] tensor
        """
        # Reshape rgb_embedding tensor to [batch_size, rgb_embedding_dim, 1]
        rgb_embedding_reshaped = rgb_embedding.unsqueeze(2)
        rgb_kv = self.rgb_kv(rgb_embedding_reshaped)
        rgb_k, rgb_v = torch.split(rgb_kv, self.text_q.out_features, dim=1)
        text_q = self.text_q(text_embedding)
        rgb_embedding = self._attn(text_q, rgb_k, rgb_v)

        return rgb_embedding


if __name__ == "__main__":
    # Define embedding dimensions
    text_embedding_dim = 1024
    rgb_embedding_dim = 1024

    # Instantiate the model
    model = CrossModalAttention(text_embedding_dim, rgb_embedding_dim, hidden_size=512)
    print(f"Model: \n{model}\n")

    # Generate random embeddings
    batch_size = 1
    text_embedding = torch.rand(batch_size, text_embedding_dim)
    rgb_embedding = torch.rand(batch_size, rgb_embedding_dim)

    # Pass embeddings through the model
    output = model(text_embedding, rgb_embedding)

    # Print output
    print(f"Output Shape: {output.shape}")
