# Copyright (c) Facebook, Inc. and its affiliates.
# This code is a modified version from its original version in the MMF repository at: mmf/mmf/trainers/mmf_trainer.py
# This trainer is modified to support compression of the model

import logging
import warnings

import omegaconf
import torch
from mmf.common.registry import registry
from mmf.datasets.multi_datamodule import MultiDataModule
from mmf.modules.metrics import Metrics
from mmf.trainers.base_trainer import BaseTrainer
from mmf.trainers.callbacks.checkpoint import CheckpointCallback
from mmf.trainers.callbacks.early_stopping import EarlyStoppingCallback
from mmf.trainers.callbacks.logistics import LogisticsCallback
from mmf.trainers.callbacks.lr_scheduler import LRSchedulerCallback
from mmf.trainers.core.callback_hook import TrainerCallbackHookMixin
from mmf.trainers.core.device import TrainerDeviceMixin
from mmf.trainers.core.evaluation_loop import TrainerEvaluationLoopMixin
from mmf.trainers.core.profiling import TrainerProfilingMixin
from mmf.trainers.core.training_loop import TrainerTrainingLoopMixin
from mmf.utils.build import build_model, build_optimizer
from mmf.utils.general import print_model_parameters
from omegaconf import DictConfig, OmegaConf
from packaging import version
import os
import pickle as pkl
torch.autograd.set_detect_anomaly(True)

from reliable_vqa.modules.decompose import Compression, get_ranks_per_layer


logger = logging.getLogger(__name__)


@registry.register_trainer("compress")
class CompressTrainer(
    TrainerCallbackHookMixin,
    TrainerTrainingLoopMixin,
    TrainerDeviceMixin,
    TrainerEvaluationLoopMixin,
    TrainerProfilingMixin,
    BaseTrainer,
):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def load(self):
        print('In Load function')
        super().load()
        self.load_fp16_scaler()

        # Callbacks
        self.on_init_start()

        self.compress_model()

        # Parallize model
        self.parallelize_model()

        # Callbacks
        self.on_init_end()

    def configure_callbacks(self):
        self.checkpoint_callback = CheckpointCallback(self.config, self)
        self.early_stop_callback = EarlyStoppingCallback(self.config, self)
        self.logistics_callback = LogisticsCallback(self.config, self)
        self.lr_scheduler_callback = LRSchedulerCallback(self.config, self)

        # Reset callbacks as they are class variables and would be shared between
        # multiple interactive shell calls to `run`
        self.callbacks = []
        # Add callbacks for execution during events
        self.callbacks.append(self.lr_scheduler_callback)
        # checkpoint_callback needs to be called after lr_scheduler_callback so that
        # lr_scheduler_callback._scheduler.step() happens before saving checkpoints
        # (otherwise the saved last_epoch in scheduler would be wrong)
        self.callbacks.append(self.checkpoint_callback)
        self.callbacks.append(self.logistics_callback)
        # Add all customized callbacks defined by users
        for callback in self.config.training.get("callbacks", []):
            callback_type = callback.type
            callback_param = callback.params
            callback_cls = registry.get_callback_class(callback_type)
            self.callbacks.append(callback_cls(self.config, self, **callback_param))

    def load_datasets(self):
        logger.info("Loading datasets")
        self.dataset_loader = MultiDataModule(self.config)
        self.train_loader = self.dataset_loader.train_dataloader()
        
        self.val_loader = self.dataset_loader.val_dataloader()
        self.test_loader = self.dataset_loader.test_dataloader()

    def load_model(self):
        logger.info("Loading model")
        if self.config.model in self.config.model_config:
            attributes = self.config.model_config[self.config.model]
        else:
            warnings.warn(
                f"Model {self.config.model}'s config not present. "
                + "Continuing with empty config"
            )
            attributes = OmegaConf.create()
        # Easy way to point to config for other model
        if isinstance(attributes, str):
            attributes = self.config.model_config[attributes]

        with omegaconf.open_dict(attributes):
            attributes.model = self.config.model

        self.model = build_model(attributes)
        print('#################')
        logger.info([p.requires_grad for p in self.model.parameters()])
        print('#################')
        
        self.model = self.model.to(self.device)

    def load_optimizer(self):
        logger.info("Loading optimizer")
        self.optimizer = build_optimizer(self.model, self.config)

    def load_metrics(self) -> None:
        logger.info("Loading metrics")
        metrics = self.config.evaluation.get("metrics", [])
        self.metrics = Metrics(metrics)
        self.metrics_params = self.metrics.required_params

    def load_fp16_scaler(self):
        if self.training_config.fp16:
            assert version.parse(torch.__version__) >= version.parse(
                "1.6"
            ), f"Using fp16 requires torch version >- 1.6, found: {torch.__version__}"
            assert self.device != torch.device("cpu"), "fp16 cannot be used on cpu"

        set_torch_grad_scaler = True
        if self.training_config.fp16 and self.distributed:
            try:
                from fairscale.optim.grad_scaler import ShardedGradScaler
                from fairscale.optim.oss import OSS

                if isinstance(self.optimizer, OSS):
                    self.scaler = ShardedGradScaler()
                    set_torch_grad_scaler = False
                    logger.info("Using FairScale ShardedGradScaler")
            except ImportError:
                logger.info("Using Pytorch AMP GradScaler")

        if set_torch_grad_scaler:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.training_config.fp16)
    
    # MODIFIED: added function to compress model's layers given in config
    def compress_model(self):
        print('in compress')
        param_names = []
        for name, param in self.model.named_parameters():
            param_names.append(name)
        self.compress = Compression()
        layers = self.config.compress.get("layers", []) # List if layer names to compress
        layers_str = ",".join(layers)
        logger.info('Layers to compress: {}'.format(layers_str) )
        ranks= self.config.compress.get("ranks", [])    # List of corresponding ranks for each layer (list of integers)
        if len(ranks) == 0:
            ratio=self.config.compress.get("ratio", 0.1)    # Ratio of compression for each layer (float between 0 and 1)
            layers,ranks = get_ranks_per_layer(self.model, ratio, layers)   # Compute low-rank factorization rank per layer, given the ratio
        
        logger.info('\n'.join([f'{l}:{r}' for l,r in zip(layers,ranks)]))
        decomposition_info = self.compress.apply_compression(self.model, layers, ranks, freeze=True)
        logger.info('Post low-rank factorization (LRF) model:\n',self.model)

        self.model = self.model.to(self.device)
        
        # reload optimizer
        #self.load_optimizer()
        new_param_names = []
        new_params = []
        for name,param in self.model.named_parameters():
            if name not in param_names:
                logger.info('New parameter: {}'.format(name))
                new_param_names.append(name)
                new_params.append(param)
        #self.optimizer.add_param_group({'params':new_params})
    #Modification END

    def train(self):
        logger.info("===== Compress Trainer =====")
        logger.info("===== Model =====")
        logger.info(self.model)
        print_model_parameters(self.model)

        if "train" in self.run_type:
            self.inference()
            self.on_train_start()
            self.training_loop()
            self.on_train_end()

        self.inference()
        self.finalize()

    def inference(self):
        logger.info(self.model)
        dataset_type = []
        if "val" in self.run_type:
            dataset_type.append("val")
        if any(rt in self.run_type for rt in ["inference", "test", "predict"]):
            dataset_type.append("test")

        for dataset in dataset_type:
            if self.config.evaluation.predict:
                self.on_prediction_start()
                self.prediction_loop(dataset)
                self.on_prediction_end()
            else:
                self.on_test_start()
                logger.info(f"Starting inference on {dataset} set")
                report, meter = self.evaluation_loop(dataset, use_tqdm=True,single_batch=False)
                self.on_test_end(report=report, meter=meter)

    def finalize(self):
        self.dataset_loader.teardown()
        self.teardown()
