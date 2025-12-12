import torch
from torch import nn


class CrossAttention(nn.Module):
    def __init__(
        self,
        x1_dim: int,
        x2_dim: int,
        num_heads: int,
        use_vis_query: bool,
        use_residual: bool,
        residual_vision: bool,
        embed_dim: int = None,
        dropout: float = 0.1,
    ) -> None:
        """
        Meant for fusion of two different modalities, x1 being language embeddings and
        x2 being visual embeddings.

        Args:
            x1_dim: Dimension of the first input (language)
            x2_dim: Dimension of the second input (visual)
            num_heads: Number of heads for the multihead attention
            use_vis_query: Whether to use visual encoding as the query and value
            use_residual: Whether to use the residual connection
            embed_dim: Dimension of the embedding space
            dropout: Dropout rate for the multihead attention
        """
        super(CrossAttention, self).__init__()

        embed_dim = embed_dim or num_heads * 64

        if x1_dim == x2_dim == embed_dim:
            self.proj1 = nn.Identity()
            self.proj2 = nn.Identity()
        else:
            # Linear layers to project x1 and x2 into embedding space
            self.proj1 = nn.Linear(x1_dim, embed_dim)
            self.proj2 = nn.Linear(x2_dim, embed_dim)

            # Initialize with Xavier initialization
            nn.init.xavier_uniform_(self.proj1.weight)
            nn.init.xavier_uniform_(self.proj2.weight)
            nn.init.zeros_(self.proj1.bias)
            nn.init.zeros_(self.proj2.bias)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # Initialize weights and biases in MultiheadAttention
        for name, param in self.multihead_attn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        self.norm = nn.LayerNorm(embed_dim)
        self.use_vis_query = use_vis_query
        self.use_residual = use_residual
        self.residual_vision = residual_vision
        self.output_size = embed_dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: [batch_size, x1_dim] tensor (language)
            x2: [batch_size, x2_dim] tensor (visual)

        Returns:
            [batch_size, embed_dim] tensor
        """

        # Project x1 and x2 into the embedding space
        x1_proj = self.proj1(x1)
        x2_proj = self.proj2(x2)

        # Reshape the tensors to be [seq_len, batch_size, embed_dim], where seq_len is 1
        x1_proj = x1_proj.unsqueeze(0)
        x2_proj = x2_proj.unsqueeze(0)

        # Perform the cross-attention calculation based on use_vis_query
        if self.use_vis_query:
            query = x2_proj
            key = x1_proj
            value = x1_proj
        else:
            query = x1_proj
            key = x2_proj
            value = x2_proj

        # output: [1, batch_size, embed_dim]
        output, _ = self.multihead_attn(query=query, key=key, value=value)

        if self.use_residual:
            # Add residual connection
            if self.residual_vision:
                output += x2_proj
            else:
                output += query

        # Apply Layer Normalization
        # output: [1, batch_size, embed_dim]
        output = self.norm(output)

        # Squeeze the sequence length dimension
        # output: [batch_size, embed_dim]
        output = output.squeeze(0)

        return output
