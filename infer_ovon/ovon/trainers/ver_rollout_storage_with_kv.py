from typing import Iterator, Optional

import torch
from habitat_baselines.common.tensor_dict import DictTree, TensorDict
from habitat_baselines.rl.ver.ver_rollout_storage import (
    VERRolloutStorage,
    generate_ver_mini_batches,
)


def build_rnn_build_seq_info(
    device: torch.device,
    episode_ids,
) -> TensorDict:
    rnn_build_seq_info = TensorDict()
    rnn_build_seq_info["episode_ids"] = (
        torch.from_numpy(episode_ids).to(device=device).reshape(-1, 1)
    )

    return rnn_build_seq_info


class VERRolloutStorageWithKVCache(VERRolloutStorage):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        max_context_length: int,
        head_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._aux_buffers["next_hidden_states"] = torch.zeros(
            self._num_envs,
            num_layers,
            2,  # key, value
            num_heads,
            max_context_length - 1,
            head_dim,
            device=self.buffers["recurrent_hidden_states"].device,
        )

        self._set_aux_buffers()

    def recurrent_generator(
        self,
        advantages: Optional[torch.Tensor],
        num_mini_batch: int,
    ) -> Iterator[DictTree]:
        """
        An exact copy of the recurrent_generator method from the parent class, the only
        difference in behaviour is:
        - The new definition of build_rnn_build_seq_info (above) is used instead.
        - "recurrent_hidden_states" is not set
        """
        if not self.variable_experience:
            yield from super().recurrent_generator(advantages, num_mini_batch)
        else:
            for mb_inds in generate_ver_mini_batches(
                num_mini_batch,
                self.sequence_lengths,
                self.num_seqs_at_step,
                self.select_inds,
                self.last_sequence_in_batch_mask,
                self.episode_ids_cpu,
            ):
                mb_inds_cpu = torch.from_numpy(mb_inds)
                mb_inds = mb_inds_cpu.to(device=self.device)

                if not self.variable_experience:
                    batch = self.buffers.map(lambda t: t.flatten(0, 1))[mb_inds]
                    if advantages is not None:
                        batch["advantages"] = advantages.flatten(0, 1)[mb_inds]
                else:
                    batch = self.buffers[mb_inds]
                    if advantages is not None:
                        batch["advantages"] = advantages[mb_inds]

                batch["rnn_build_seq_info"] = build_rnn_build_seq_info(
                    device=self.device,
                    episode_ids=self.episode_ids_cpu[mb_inds_cpu],
                )

                yield batch.to_tree()
