from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import transformers
from habitat import logger

import math
from transformers import LlamaConfig, LlamaModel
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaAttention,
)


class TransformerEncoder(nn.Module):
    def __init__(self, input_size: int, config):
        super().__init__()
        self.model_name = config.model_name
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.n_hidden = config.n_hidden
        self.n_mlp_hidden = config.n_mlp_hidden
        self.max_context_length = config.max_context_length
        self.max_position_embeddings = config.max_position_embeddings
        self.shuffle_pos_id_for_update = config.shuffle_pos_id_for_update

        self.feats_proj = nn.Linear(input_size, self.n_hidden)

        if self.model_name == "gpt":
            self.hf_config = transformers.GPT2Config(
                vocab_size=0,
                n_embd=self.n_hidden,
                n_layer=self.n_layers,
                n_head=self.n_heads,
            )
            self.model = transformers.GPT2Model(self.hf_config)
            self.model.wte.weight.requires_grad_(False)
        elif self.model_name == "llama":
            self.hf_config = LlamaConfig(
                hidden_size=self.n_hidden,
                intermediate_size=self.n_mlp_hidden,
                num_hidden_layers=self.n_layers,
                num_attention_heads=self.n_heads,
                max_position_embeddings=self.max_position_embeddings,
            )
            self.model = LlamaModel(self.hf_config)
            for param in self.model.embed_tokens.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Unrecognized {self.model_name}")

        self._expected_cache_shape = (
            self.n_layers,
            2,
            self.n_heads,
            self.max_context_length - 1,
            self.n_hidden // self.n_heads,
        )

        logger.info(f"Done loading {self.model_name} transformer.")

    def forward(
        self,
        feats,
        rnn_hidden_states,
        masks,
        rnn_build_seq_info,
        **kwargs,
    ):
        feats = self.feats_proj(feats)
        gpu_device = feats.device

        # Assume update only occurs when episode_ids is injected by dagger.py or ppo.py
        is_update_batch = "episode_ids" in rnn_build_seq_info

        if is_update_batch:
            episode_ids = rnn_build_seq_info["episode_ids"].reshape(-1)

            # Split the input sequences into episodes
            assert feats.shape[0] == episode_ids.shape[0]
            all_sequences = split_by_id(feats, episode_ids)
            all_sequences = split_longer_seqs(
                all_sequences, max_len=self.max_context_length
            )

            # Pad the shorter sequences with 0s, and create the attention mask to
            # keep track of the padded elements
            feats, attention_mask = pad_sequences(all_sequences, device=gpu_device)

            # Create the position_ids tensor
            batch_size, seq_len = feats.shape[:2]
            if self.shuffle_pos_id_for_update:
                position_ids = rand_shifted_position_ids(
                    self.max_position_embeddings - 1,
                    batch_size,
                    seq_len,
                    device=gpu_device,
                )
            else:
                position_ids = split_by_id(rnn_build_seq_info["step_id"], episode_ids)
                position_ids = split_longer_seqs(
                    position_ids, max_len=self.max_context_length
                )
                position_ids, _ = pad_sequences(position_ids, device=gpu_device)
                position_ids = position_ids.squeeze(1)

            # Send the tensors to the GPU
            feats = feats.to(gpu_device)
            position_ids = position_ids.to(gpu_device)
            attention_mask = attention_mask.to(gpu_device)

            output = self.model(
                inputs_embeds=feats,
                position_ids=position_ids,
                attention_mask=attention_mask.to(gpu_device),
            )
            feats = extract_hidden_features(output.last_hidden_state, attention_mask)
        else:
            # Assuming that we are doing online inference
            assert feats.ndim == 2  # (batch_size, embed_dim)

            if rnn_hidden_states.shape[1:] != self._expected_cache_shape:
                print(
                    "WARNING: Received unexpected shape for kv_cache. This is OK if you"
                    " are doing checkpoint evaluation. Reshaping to expected shape."
                )
                rnn_hidden_states = torch.zeros(
                    rnn_hidden_states.shape[0],
                    *self._expected_cache_shape,
                    device=gpu_device,
                )

            feats = feats.unsqueeze(1)  # (batch_size, 1, embed_dim), seq_len of 1 step

            # Create attention mask
            step_ids = rnn_build_seq_info["step_id"].reshape(-1)
            attention_mask = create_mask_with_trailing_ones(
                step_ids + 1, S=self.max_context_length
            )

            output = self.model(
                inputs_embeds=feats,
                position_ids=step_ids,
                attention_mask=attention_mask,
                past_key_values=tensor_to_kv_cache(rnn_hidden_states),
            )

            cache = truncate_cache(
                output.past_key_values, max_len=self.max_context_length - 1
            )
            rnn_hidden_states = torch.stack(
                [torch.stack([k, v], dim=1) for k, v in cache], dim=1
            )
            feats = output.last_hidden_state[:, -1, :].squeeze(1)

        return feats, rnn_hidden_states

    @property
    def num_recurrent_layers(self):
        return self.n_layers


def tensor_to_kv_cache(
    kv_tensor: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Convert a tensor representation of key-value cache back to its original tuple
    format.

    Args:
        kv_tensor (torch.Tensor): A tensor representation of the key-value cache.

    Returns:
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...]: The original tuple format of
            key-value cache placeholders.
    """
    # Infer dimensions from the input kv_tensor
    batch_size, num_layers, _, num_heads, seq_len, head_dim = kv_tensor.shape

    kv_cache = tuple(
        (kv_tensor[:, layer_idx, 0, :, :, :], kv_tensor[:, layer_idx, 1, :, :, :])
        for layer_idx in range(num_layers)
    )
    return kv_cache


def pad_sequences(
    sequences: List[torch.Tensor],
    padding_value: float = 0.0,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads all sequences to be the same length as the longest given sequence, and also
    returns the corresponding attention_mask to be used by the transformer.

    Args:
        sequences (List[torch.Tensor]): A list of sequences to be padded.
            padding_value (float): The value to be used for padding.
        padding_value (float): The value to be used for padding.
        device (torch.device): The device to be used for the padded sequences and

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the padded sequences and
            the corresponding attention mask.
            - padded_sequences (torch.Tensor): The padded sequences of shape
                (batch_size, seq_len), where seq_len is the length of the longest given
                sequence.
            - attention_mask (torch.Tensor): The attention mask of shape (batch_size,
                seq_len).
    """
    max_length = max(seq.size(0) for seq in sequences)
    padded_sequences = torch.full(
        (len(sequences), max_length, sequences[0].size(-1)),
        padding_value,
        device=device,
    )
    attention_mask = torch.zeros(len(sequences), max_length, device=device)

    for i, seq in enumerate(sequences):
        end = seq.size(0)
        padded_sequences[i, :end, :] = seq
        attention_mask[i, :end] = 1

    return padded_sequences, attention_mask


def extract_hidden_features(
    padded_output_embeds: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Filters out the hidden features of the padded output embeddings using the attention
    mask. The attention mask has a value of -inf if the corresponding time step is
    padded, and 0 otherwise.

    Args:
        padded_output_embeds (torch.Tensor): The padded output embeddings of shape
            (batch_size, seq_len, embed_dim).
        attention_mask (torch.Tensor): The attention mask of shape (batch_size,
            seq_len).
    Returns:
        torch.Tensor: The hidden features of the padded output embeddings of shape
            (new_batch_size, embed_dim), where new_batch_size is the number of
            non-padded time steps across all sequences in the batch.
    """
    # Flatten the attention mask to identify all non-padded elements
    non_padded_indices = torch.nonzero(attention_mask.flatten().cpu(), as_tuple=True)[0]

    # Flatten the embeddings tensor and select only the non-padded elements
    flattened_embeddings = padded_output_embeds.reshape(
        -1, padded_output_embeds.size(-1)
    )
    filtered_features = flattened_embeddings.index_select(
        0, non_padded_indices.to(flattened_embeddings.device)
    )

    return filtered_features


def rand_shifted_position_ids(
    max_position_id: int, batch_size: int, seq_len: int, device: torch.device = None
) -> torch.Tensor:
    """
    Given a shape (batch_size, seq_len), returns a new long tensor of shape
    (batch_size, seq_len). The returned tensor has the following properties:
    1. Values are uniformly sampled randomly, but all elements must be between 0 and
    max_position_id.
    2. Each column must have a value equal to the previous column plus 1.


    Args:
        max_position_id (int): The maximum value that the last index can take.
        batch_size (int): The batch size.
        seq_len (int): The sequence length.
        device (torch.device): The device to be used for the returned tensor.

    Returns:
        torch.Tensor: A random sequence of position ids of shape (batch_size, seq_len).
    """
    # Ensure the sequence can fit within the max_position_id
    if seq_len - 1 > max_position_id:
        raise ValueError("Sequence length is too large to fit within max_position_id")

    # Randomly select the start position for each sequence in the batch
    start_positions = torch.randint(
        low=0, high=max_position_id - seq_len + 2, size=(batch_size,), device=device
    )

    # Create a range tensor from 0 to seq_len - 1
    range_tensor = torch.arange(seq_len, device=device)

    # Add the start position to each column, ensuring it fits within max_position_id
    shifted_positions = (start_positions.unsqueeze(1) + range_tensor.unsqueeze(0)) % (
        max_position_id + 1
    )

    return shifted_positions


def truncate_cache(
    kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], max_len: int
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Remove the oldest entries from each key-value pair in the cache so that the total
    number of columns does not exceed max_len.

    Args:
        kv_cache (Tuple[Tuple[torch.Tensor, torch.Tensor], ...]): The key-value cache
        from each layer and each head of the transformer model.
        max_len (int): The maximum number of columns to retain in each tensor.

    Returns:
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...]: Updated key-value cache with
        the oldest entries removed to ensure the length is at most max_len.
    """
    updated_cache = tuple(
        (
            (
                k[:, :, -max_len:, :],
                v[:, :, -max_len:, :],
            )  # Keep the last max_len columns
            for k, v in kv_cache
        )
    )
    return updated_cache


def split_longer_seqs(seqs: List[torch.Tensor], max_len: int) -> List[torch.Tensor]:
    """
    Splits any long sequences in a list of sequences into multiple sequences of length
    max_len (except for the last sequence, which may be shorter). Returns a list of
    sequences and a list of lists of indices so that the original sequences can be
    recovered.

    Args:
        seqs (List[torch.Tensor]): A list of sequences, each with a shape (batch_size,
            seq_len).
        max_len (int): The maximum length to truncate to.

    Returns:
        List[torch.Tensor]: A list of split sequences.
    """
    """
    Splits any long sequences in a list of sequences into multiple sequences of length
    max_len (except for the last sequence, which may be shorter). Returns a list of
    sequences.

    Args:
        seqs (List[torch.Tensor]): A list of sequences, each with a shape (seq_len,
            embed_dim).
        max_len (int): The maximum length to truncate to.

    Returns:
        List[torch.Tensor]: A list of split sequences.
    """
    split_seqs = []  # List to hold the split sequences

    for seq in seqs:
        seq_len = seq.shape[0]
        if seq_len <= max_len:
            # If the sequence length is within the limit, add it directly
            split_seqs.append(seq)
        else:
            # Split the sequence into chunks of max_len
            for i in range(0, seq_len, max_len):
                end_idx = min(i + max_len, seq_len)
                split_seqs.append(seq[i:end_idx, :])

    return split_seqs


def split_by_id(sequence: torch.Tensor, seq_ids: torch.Tensor) -> List[torch.Tensor]:
    """
    Splits a sequence into chunks based on the sequence ids.

    Args:
        sequence (torch.Tensor): The sequence to split. Shape: (batch_size, embed_dim).
        seq_ids (torch.Tensor): The sequence ids. Shape: (batch_size,).

    Returns:
        List[torch.Tensor]: A list of split sequences.
    """
    unique_ids, counts = torch.unique_consecutive(seq_ids, return_counts=True)

    # Loop through the ordered unique sequence ids
    split_sequences = []
    start_idx = 0
    for uid, c in zip(unique_ids, counts):
        split_sequences.append(sequence[start_idx : start_idx + c])
        start_idx += c

    return split_sequences


def create_mask_with_trailing_ones(
    num_trailing_ones: torch.Tensor, S: int
) -> torch.Tensor:
    """
    Creates a 2D mask where each row has a specified number of trailing ones.

    The number of trailing ones in each row of the mask is determined by
    the corresponding value in the input tensor. The rest of the elements
    in each row are zeros.

    Args:
        num_trailing_ones (Tensor): A 1D tensor of shape (N,) containing values in the
            range [0, S-1]. Each value 'n' in this tensor specifies the number of
            trailing ones in the corresponding row of the output mask.
        S (int): The size of the second dimension of the output mask. It also represents
            the maximum number of trailing ones that can appear in any row of the mask.

    Returns:
        torch.Tensor: A 2D mask of shape (N, S) where each row contains zeros followed
            by a number of ones as specified by the corresponding element in the input
            tensor.

    Example:
    >>> tensor = torch.tensor([0, 8, 5, 4, 2])
    >>> create_mask_with_trailing_ones(tensor, 10)
    tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.]])
    """
    # Create a 2D grid where each row is 0, 1, ..., S-1
    grid = torch.arange(S, device=num_trailing_ones.device).expand(
        len(num_trailing_ones), S
    )

    # Reshape the input tensor for broadcasting
    reshaped_tensor = num_trailing_ones.view(-1, 1)

    # Create the mask with trailing ones
    mask = (grid >= (S - reshaped_tensor)).float()

    return mask


def patched_llama_attention_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    ##### START OF PATCHED CODE #####
    if position_ids is None:
        rotary_emb_seq_len = kv_seq_len
    else:
        rotary_emb_seq_len = torch.max(position_ids).item() + 1
    cos, sin = self.rotary_emb(value_states, seq_len=rotary_emb_seq_len)
    ##### END OF PATCHED CODE #####

    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights,
            torch.tensor(
                torch.finfo(attn_weights.dtype).min, device=attn_weights.device
            ),
        )

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# Apply rotary position embedding patch
LlamaAttention.forward = patched_llama_attention_forward
