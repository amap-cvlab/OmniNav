# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import os
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import EvalPrediction
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available

from swift.utils import JsonlWriter, Serializer, gc_collect, get_logger, unwrap_model_for_generation
from .arguments import Seq2SeqTrainingArguments, TrainingArguments
from .mixin import DataLoaderMixin, SwiftMixin

logger = get_logger()


class Trainer(SwiftMixin, HfTrainer):
    args: TrainingArguments

    @contextmanager
    def _patch_loss_function(self):
        model = self.model
        if isinstance(model, PeftModel):
            model = model.model
        model_cls = model.__class__
        if not hasattr(model_cls, 'loss_function'):
            yield
            return

        loss_function = model.loss_function
        _old_loss_function = model_cls.loss_function

        @staticmethod
        @wraps(loss_function)
        def new_loss_function(logits, labels, **kwargs):
            labels = labels.to(logits.device)  # fix device_map
            return loss_function(logits=logits, labels=labels, **kwargs)

        model_cls.loss_function = new_loss_function
        try:
            yield
        finally:
            model_cls.loss_function = _old_loss_function

    def train(self, *args, **kwargs):
        with self._patch_loss_function():
            return super().train(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if inputs.get('labels') is not None:
            self._compute_acc(outputs, inputs['labels'])
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss


class EmbeddingTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.calculate_metric
        self.preprocess_logits_for_metrics = None
        self.label_names = ['labels']

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        from swift.plugin.loss import infonce_loss, calculate_paired_metrics, calculate_infonce_metrics
        if self.compute_loss_func is infonce_loss:
            return calculate_infonce_metrics(eval_prediction.predictions, eval_prediction.label_ids)
        else:
            return calculate_paired_metrics(eval_prediction.predictions, eval_prediction.label_ids)


class RerankerTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.calculate_metric
        self.label_names = ['labels']

        # Set up preprocess_logits_for_metrics to reduce memory usage for generative reranker
        from swift.plugin.loss import get_loss_func, LossType
        if self.compute_loss_func in [
                get_loss_func(LossType.generative_reranker),
                get_loss_func(LossType.listwise_generative_reranker)
        ]:
            self.preprocess_logits_for_metrics = self._preprocess_generative_reranker_logits
        else:
            self.preprocess_logits_for_metrics = None

    def _preprocess_generative_reranker_logits(self, logits, labels):
        """
        Preprocess logits for generative reranker to reduce memory usage.
        Extract only the yes/no token logits instead of keeping the full vocab logits.
        """
        import torch
        import os

        # Get token IDs for positive and negative tokens
        positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
        negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')

        tokenizer = getattr(self, 'processing_class', None)
        if tokenizer is None:
            # Fallback: return full logits if tokenizer not available
            return logits

        try:
            positive_token_id = tokenizer.convert_tokens_to_ids(positive_token)
            negative_token_id = tokenizer.convert_tokens_to_ids(negative_token)
        except Exception:
            # Fallback: return full logits if token conversion fails
            return logits

        # Extract only the yes/no token logits from the last position
        # This dramatically reduces memory usage
        if len(logits.shape) == 3:
            # Extract directly from last position: [batch_size, seq_len, vocab_size] -> [batch_size, 2]
            positive_logits = logits[:, -1, positive_token_id]  # [batch_size]
            negative_logits = logits[:, -1, negative_token_id]  # [batch_size]
            # Return as [batch_size, 2] tensor instead of full [batch_size, seq_len, vocab_size]
            logits = torch.stack([negative_logits, positive_logits], dim=1)
            return logits
        else:
            # Unexpected shape, return as-is
            return logits

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        from swift.plugin.loss import (get_loss_func, LossType, calculate_reranker_metrics)

        # Check if we're using generative reranker (point-wise or list-wise)
        if self.compute_loss_func in [
                get_loss_func(LossType.generative_reranker),
                get_loss_func(LossType.listwise_generative_reranker)
        ]:
            # For generative reranker, predictions are now [batch_size, 2] from preprocessing
            # We need to handle this differently
            predictions = eval_prediction.predictions
            if len(predictions.shape) == 2 and predictions.shape[1] == 2:
                # Predictions are already preprocessed [batch_size, 2] format
                # Apply softmax to get probabilities
                import numpy as np
                exp_logits = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
                probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                relevance_scores = probabilities[:, 1]  # Positive class probability
                return calculate_reranker_metrics(relevance_scores, eval_prediction.label_ids)
            else:
                # Fallback to original method if preprocessing didn't work
                raise ValueError('Unexpected predictions shape')
        else:
            # For standard reranker (point-wise or list-wise)
            return calculate_reranker_metrics(eval_prediction.predictions, eval_prediction.label_ids)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Check if we have a custom loss function
        if self.compute_loss_func is not None:
            from swift.plugin.loss import get_loss_func, LossType
            loss_kwargs = {}

            if self.compute_loss_func in [
                    get_loss_func(LossType.generative_reranker),
                    get_loss_func(LossType.listwise_generative_reranker)
            ]:
                loss_kwargs['trainer'] = self

            # Get labels and compute outputs
            labels = inputs.get('labels')
            if labels is not None:
                labels = inputs.pop('labels')

            outputs = model(**inputs)

            if labels is not None:
                # Call custom loss function
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, **loss_kwargs)
            else:
                # Fallback to model's loss
                loss = outputs.loss

            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                loss = loss / self.args.gradient_accumulation_steps

            if labels is not None:
                self._compute_acc(outputs, labels)

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

from torch.utils.data import ConcatDataset
import torch
import random

class PureBatchSampler(torch.utils.data.Sampler):
    def __init__(self, datasets, batch_size, probs=None, drop_last=True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.lengths = [len(ds) for ds in datasets]
        self.offsets = [0]
        for l in self.lengths:
            self.offsets.append(self.offsets[-1] + l)
        self.probs = probs or [l/sum(self.lengths) for l in self.lengths]
        
    def __iter__(self):
        sub_perm = [torch.randperm(l).tolist() for l in self.lengths]
        sub_ptrs = [0] * len(self.lengths)
        while True:
            # 只从剩余能采batch的池中选择
            available = [
                i for i, ptr in enumerate(sub_ptrs)
                if ptr + self.batch_size <= self.lengths[i]
            ]
            if not available:
                break
            probs_now = [self.probs[i] for i in available]
            idx = random.choices(available, weights=probs_now)[0]
            indices = sub_perm[idx][sub_ptrs[idx]:sub_ptrs[idx]+self.batch_size]
            sub_ptrs[idx] += self.batch_size
            yield [self.offsets[idx] + i for i in indices]

    def __len__(self):
        return sum(l // self.batch_size for l in self.lengths)


import torch
from torch.utils.data import Sampler
from itertools import accumulate
import torch.distributed as dist

class FixedProportionSampler(Sampler):
    """
    支持 DDP 的非流式采样器，保证：
    - 每个 epoch 按数据量比例从各数据集采样
    - 每个 batch 来自同一数据集（pure batch）
    - 所有 batch 全局 shuffle
    - DDP 下各 rank 无重复
    """
    def __init__(
        self,
        datasets,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False
    ):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        self.lengths = [len(ds) for ds in datasets]
        self.offsets = list(accumulate([0] + self.lengths))  # 0, len0, len0+len1, ...

        if drop_last:
            self.batches_per_dataset = [l // batch_size for l in self.lengths]
        else:
            self.batches_per_dataset = [(l + batch_size - 1) // batch_size for l in self.lengths]

        self.epoch = 0  # 默认 epoch

    def __iter__(self):
        generator = torch.Generator()
        seed = self.seed + self.epoch
        generator.manual_seed(seed)

        all_batch_indices = []

        # 1. 为每个数据集生成 batch
        for i, dataset_len in enumerate(self.lengths):
            n_batches = self.batches_per_dataset[i]
            if n_batches == 0:
                continue

            indices = torch.randperm(dataset_len, generator=generator).tolist()

            # 填充不足 batch 的情况（仅当 drop_last=False）
            required = n_batches * self.batch_size
            if len(indices) < required:
                extended = []
                while len(extended) < required:
                    extended.extend(indices)
                indices = extended[:required]

            # 切成 batch 并加偏移
            for j in range(n_batches):
                start = j * self.batch_size
                end = start + self.batch_size
                batch = [self.offsets[i] + indices[k] for k in range(start, end)]
                all_batch_indices.append(batch)

        # 2. 全局 shuffle
        if self.shuffle:
            order = torch.randperm(len(all_batch_indices), generator=generator).tolist()
            all_batch_indices = [all_batch_indices[i] for i in order]

        # 3. DDP 分片：每个 rank 只取一部分，避免重复
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            all_batch_indices = all_batch_indices[rank::world_size]

            # 重要：如果分片后为空，至少返回一个 batch（避免 DataLoader 报错）
            if len(all_batch_indices) == 0 and len(all_batch_indices) > 0:
                all_batch_indices = [all_batch_indices[0]]

        return iter(all_batch_indices)

    def __len__(self):
        total = sum(self.batches_per_dataset)
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            return total // world_size
        return total

    def set_epoch(self, epoch: int):
        """由 Trainer 自动调用"""
        self.epoch = epoch


class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    args: Seq2SeqTrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        if self.args.predict_with_generate:
            from swift.llm import PtEngine
            self.infer_engine = PtEngine.from_model_template(
                self.model, self.template, max_batch_size=self.args.per_device_eval_batch_size)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'predict.jsonl'))
    #mark add
    # def get_train_dataloader(self):
        # 直接粘贴你自定义的逻辑，
        # 仅处理train_dataset是list的情况，否则用super默认逻辑
        # if isinstance(self.train_dataset, list):
        #     # --------- 下面是你按比例采样的实现 -------------
        #     import random
        #     from torch.utils.data import DataLoader
        #     batch_size = self._train_batch_size
        #     args = self.args
        #     datasets = self.train_dataset
        #     dataloaders = [
        #         DataLoader(
        #             ds,
        #             batch_size=batch_size,
        #             collate_fn=self.data_collator,
        #             num_workers=args.dataloader_num_workers,
        #             pin_memory=args.dataloader_pin_memory,
        #             persistent_workers=args.dataloader_persistent_workers,
        #             prefetch_factor=args.dataloader_prefetch_factor,
        #             drop_last=args.dataloader_drop_last,
        #             shuffle=args.train_dataloader_shuffle,
        #         ) for ds in datasets
        #     ]
        #     lengths = [len(ds) for ds in datasets]
        #     probs = [l/sum(lengths) for l in lengths]
        #     def mixed_gen():
        #         loader_iters = [iter(dl) for dl in dataloaders]
        #         exhausted = [False]*len(dataloaders)
        #         while not all(exhausted):
        #             idx = random.choices(range(len(dataloaders)), weights=probs)[0]
        #             if exhausted[idx]:
        #                 continue
        #             try:
        #                 batch = next(loader_iters[idx])
        #             except StopIteration:
        #                 exhausted[idx] = True
        #                 continue
        #             yield batch
        #     class MixedLoader:
        #         def __iter__(self_): return mixed_gen()
        #         def __len__(self_): return sum(len(dl) for dl in dataloaders)
        #     return MixedLoader()
                # -------------------------------------------------
        # else:
        #     # 单一数据集直接走 Mixin 或父类默认实现
        #     return super().get_train_dataloader()
    # def get_train_dataloader(self):
    #     if isinstance(self.train_dataset, list):
    #         batch_size = self._train_batch_size
    #         datasets = self.train_dataset
    #         lengths = [len(ds) for ds in datasets]
    #         probs = [l/sum(lengths) for l in lengths]
    #         ds = ConcatDataset(datasets)
    #         batch_sampler = PureBatchSampler(datasets, batch_size, probs=probs, drop_last=self.args.dataloader_drop_last)
    #         return torch.utils.data.DataLoader(
    #             ds,
    #             batch_sampler=batch_sampler,
    #             collate_fn=self.data_collator,
    #             num_workers=self.args.dataloader_num_workers,
    #             pin_memory=self.args.dataloader_pin_memory,
    #             persistent_workers=self.args.dataloader_persistent_workers,
    #             prefetch_factor=self.args.dataloader_prefetch_factor,
    #         )
    #     else:
    #         return super().get_train_dataloader()
    def get_train_dataloader(self):
        if isinstance(self.train_dataset, list):
            batch_size = self._train_batch_size
            datasets = self.train_dataset
            concat_dataset = ConcatDataset(datasets)

            sampler = FixedProportionSampler(
                datasets=datasets,
                batch_size=batch_size,
                shuffle=self.args.train_dataloader_shuffle,
                seed=self.args.seed,
                drop_last=self.args.dataloader_drop_last,
            )

            return torch.utils.data.DataLoader(
                concat_dataset,
                batch_sampler=sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=self.args.dataloader_persistent_workers,
                prefetch_factor=self.args.dataloader_prefetch_factor,
            )
        else:
            return super().get_train_dataloader()

    #mark over
    @staticmethod
    def _predict_data_collator(batch):
        return {'_data': batch}

    @contextmanager
    def _patch_predict_with_generate(self):
        origin_data_collator = self.data_collator
        self.data_collator = self._predict_data_collator
        try:
            yield
        finally:
            self.data_collator = origin_data_collator

    def evaluate(self, *args, **kwargs):
        context = self._patch_predict_with_generate() if self.args.predict_with_generate else nullcontext()
        with context:
            res = super().evaluate(*args, **kwargs)
            gc_collect()
            return res

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        from swift.llm import RequestConfig, InferRequest
        data_list = inputs['_data']
        labels_list = [InferRequest.remove_response(data['messages']) for data in data_list]
        with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation), self.template.generate_context():
            resp_list = self.infer_engine.infer(
                data_list,
                RequestConfig(max_tokens=self.model.generation_config.max_new_tokens),
                use_tqdm=False,
                template=self.template)

        response_list = []
        jsonl_cache = []
        device = self.args.device
        for data, resp, labels in zip(data_list, resp_list, labels_list):
            response = resp.choices[0].message.content
            jsonl_cache.append({'response': response, 'labels': labels, **data})
            response_list.append(Serializer.to_tensor(resp.choices[0].message.content).to(device=device))
        self.jsonl_writer.append(jsonl_cache, gather_obj=True)
        labels_list = [Serializer.to_tensor(labels).to(device=device) for labels in labels_list]
        response_list = pad_sequence(response_list, batch_first=True, padding_value=0)
        labels_list = pad_sequence(labels_list, batch_first=True, padding_value=0)
        return None, response_list, labels_list

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        from swift.plugin.loss import get_loss_func
        loss_kwargs = {}
        labels = None
        compute_loss_func = self.compute_loss_func
        loss_scale = inputs.pop('loss_scale', None)
        if loss_scale is not None:
            loss_kwargs['loss_scale'] = loss_scale
            if compute_loss_func is None:
                compute_loss_func = get_loss_func('loss_scale')

        sample_channels = inputs.pop('channel', None)
        if sample_channels is not None and self.args.channels is not None:
            state = self.state
            setattr(state, 'local_step', getattr(state, 'local_step', 0))
            setattr(state, 'ch_loss_steps', getattr(state, 'ch_loss_steps', {}))

            loss_kwargs['sample_channels'] = sample_channels
            loss_kwargs['trainer'] = self
            if inputs.get('position_ids') is not None:
                loss_kwargs['position_ids'] = inputs['position_ids']

        if (self.label_smoother is not None or compute_loss_func is not None) and 'labels' in inputs:
            labels = inputs.pop('labels')

        use_logits_to_keep = self.get_use_logits_to_keep('labels' in inputs)
        if use_logits_to_keep:
            inputs['labels'], logits_to_keep = self.get_logits_to_keep(inputs['labels'])
            if logits_to_keep is not None:
                inputs['logits_to_keep'] = logits_to_keep
        with self.template.compute_loss_context(self.model, inputs):
            outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs['labels']
            outputs.loss = outputs.loss.to(labels.device)
            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                outputs.loss = outputs.loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if compute_loss_func is not None:
                loss = compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, **loss_kwargs)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)

        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            loss = sequence_parallel.reduce_outputs(loss, labels)

        if getattr(self.args, 'average_tokens_across_devices', False) and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        if outputs.logits is not None and labels is not None and not return_outputs:
            # Liger does not have logits
            self._compute_acc(outputs, labels)
        return (loss, outputs) if return_outputs else loss
