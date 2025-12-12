from multiprocessing.context import BaseContext

from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ver.inference_worker import (
    InferenceWorker,
    InferenceWorkerProcess,
)
from habitat_baselines.rl.ver.worker_common import WorkerBase


class InferenceWorkerWithKVProcess(InferenceWorkerProcess):
    def _update_storage_no_ver(
        self, prev_step: TensorDict, current_step: TensorDict, current_steps
    ):
        current_step.pop("recurrent_hidden_states")
        super()._update_storage_no_ver(prev_step, current_step, current_steps)

    def _update_storage_ver(
        self, prev_step: TensorDict, current_step: TensorDict, my_slice
    ):
        current_step.pop("recurrent_hidden_states")
        super()._update_storage_ver(prev_step, current_step, my_slice)


class InferenceWorkerWithKV(InferenceWorker):
    def __init__(self, mp_ctx: BaseContext, use_kv: bool = True, *args, **kwargs):
        if use_kv:
            self.setup_queue = mp_ctx.SimpleQueue()
            WorkerBase.__init__(
                self,
                mp_ctx,
                InferenceWorkerWithKVProcess,
                self.setup_queue,
                *args,
                **kwargs,
            )
        else:
            super().__init__(mp_ctx, *args, **kwargs)
