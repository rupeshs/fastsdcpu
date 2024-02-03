"""
The Trainer class, to easily train a super-image model from scratch.
The design is inspired by the HuggingFace transformers library at
https://github.com/huggingface/transformers/.
"""

import os
import copy
import logging
from typing import Optional, Union, Dict, Callable

from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from .modeling_utils import PreTrainedModel
from .configuration_utils import PretrainedConfig
from .file_utils import (
    WEIGHTS_NAME,
    WEIGHTS_NAME_SCALE,
    CONFIG_NAME
)
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalPrediction,
    set_seed
)
from .training_args import TrainingArguments
from .utils.metrics import AverageMeter, compute_metrics

logger = logging.getLogger(__name__)


def make_optimizer(opt, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    optimizer_function = optim.Adam
    kwargs = {'betas': (opt.beta1, opt.beta2), 'eps': opt.epsilon, 'lr': opt.learning_rate,
              'weight_decay': opt.gamma}

    return optimizer_function(trainable, **kwargs)


def make_dual_optimizer(opt, dual_models):
    dual_optimizers = []
    for dual_model in dual_models:
        temp_dual_optim = torch.optim.Adam(
            params=dual_model.parameters(),
            lr=opt.learning_rate,
            betas=(opt.beta1, opt.beta2),
            eps=opt.epsilon,
            weight_decay=opt.gamma)
        dual_optimizers.append(temp_dual_optim)

    return dual_optimizers


def make_scheduler(opt, my_optimizer):
    scheduler = lrs.CosineAnnealingLR(
        my_optimizer,
        float(opt.num_train_epochs),
        eta_min=opt.eta_min
    )

    return scheduler


def make_dual_scheduler(opt, dual_optimizers):
    dual_scheduler = []
    for i in range(len(dual_optimizers)):
        scheduler = lrs.CosineAnnealingLR(
            dual_optimizers[i],
            float(opt.num_train_epochs),
            eta_min=opt.eta_min
        )
        dual_scheduler.append(scheduler)

    return dual_scheduler


class TrainerDrn:
    """
    Trainer is a simple class implementing the training and eval loop for PyTorch to train a super-image model.
    Args:
        model (:class:`~super_image.PreTrainedModel` or :obj:`torch.nn.Module`, `optional`):
            The model to train, evaluate or use for predictions. If not provided, a ``model_init`` must be passed.
            .. note::
                :class:`~super_image.Trainer` is optimized to work with the :class:`~super_image.PreTrainedModel`
                provided by the library. You can still use your own models defined as :obj:`torch.nn.Module` as long as
                they work the same way as the super_image models.
        args (:class:`~super_image.TrainingArguments`, `optional`):
            The arguments to tweak for training. Will default to a basic instance of
            :class:`~super_image.TrainingArguments` with the ``output_dir`` set to a directory named `tmp_trainer` in
            the current directory if not provided.
        train_dataset (:obj:`torch.utils.data.dataset.Dataset` or :obj:`torch.utils.data.dataset.IterableDataset`):
            The dataset to use for training.
        eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
             The dataset to use for evaluation.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        train_dataset: Dataset = None,
        eval_dataset: Optional[Dataset] = None,
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)
        self.args = args
        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)

        if model is None:
            raise RuntimeError("`Trainer` requires a `model`")

        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.best_epoch = 0
        self.best_metric = 0.0
        args.epsilon = 1e-8
        args.beta1 = 0.9
        args.beta2 = 0.999
        args.eta_min = 1e-7
        self.optimizer = make_optimizer(args, self.model)
        self.scheduler = make_scheduler(args, self.optimizer)
        self.dual_models = self.model.dual_models
        self.dual_optimizers = make_dual_optimizer(args, self.dual_models)
        self.dual_scheduler = make_dual_scheduler(args, self.dual_optimizers)
        self.error_last = 1e8

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            **kwargs,
    ):
        """
        Main training entry point.
        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~super_image.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~super_image.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        args = self.args

        epochs_trained = 0
        device = args.device
        num_train_epochs = args.num_train_epochs
        learning_rate = args.learning_rate
        train_batch_size = args.train_batch_size
        train_dataset = self.train_dataset
        train_dataloader = self.get_train_dataloader()
        step_size = int(len(train_dataset) / train_batch_size * 200)

        # # Load potential model checkpoint
        # if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
        #     resume_from_checkpoint = get_last_checkpoint(args.output_dir)
        #     if resume_from_checkpoint is None:
        #         raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")
        #
        # if resume_from_checkpoint is not None:
        #     if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
        #         raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")
        #
        #     logger.info(f"Loading model from {resume_from_checkpoint}).")
        #
        #     if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
        #         config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
        #
        #     state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
        #     # If the model is on the GPU, it still works!
        #     self._load_state_dict_in_model(state_dict)
        #
        #     # release memory
        #     del state_dict

        for epoch in range(epochs_trained, num_train_epochs):
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = learning_rate * (0.1 ** (epoch // int(num_train_epochs * 0.8)))

            self.model.train()
            epoch_losses = AverageMeter()

            with tqdm(total=(len(train_dataset) - len(train_dataset) % train_batch_size)) as t:
                t.set_description(f'epoch: {epoch}/{num_train_epochs - 1}')

                for data in train_dataloader:
                    lr, hr = data

                    lr = lr.to(device)
                    hr = hr.to(device)

                    self.optimizer.zero_grad()

                    for i in range(len(self.dual_optimizers)):
                        self.dual_optimizers[i].zero_grad()

                    # forward
                    sr = self.model(lr[0])
                    sr2lr = []
                    for i in range(len(self.dual_models)):
                        sr2lr_i = self.dual_models[i](sr[i - len(self.dual_models)])
                        sr2lr.append(sr2lr_i)

                    # compute primary loss
                    loss_primary = self.loss(sr[-1], hr)
                    for i in range(1, len(sr)):
                        loss_primary += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])

                    # compute dual loss
                    loss_dual = self.loss(sr2lr[0], lr[0])
                    for i in range(1, len(self.scale)):
                        loss_dual += self.loss(sr2lr[i], lr[i])

                    # compute total loss
                    loss = loss_primary + self.opt.dual_weight * loss_dual

                    if loss.item() < self.opt.skip_threshold * self.error_last:
                        loss.backward()
                        self.optimizer.step()
                        for i in range(len(self.dual_optimizers)):
                            self.dual_optimizers[i].step()

            self.error_last = self.loss.log[-1, -1]
            self.scheduler.step()
            for i in range(len(self.dual_scheduler)):
                self.dual_scheduler[i].step()

            self.eval(epoch)

    def eval(self, epoch):
        args = self.args

        scale = self.model.config.scale
        device = args.device
        eval_dataloader = self.get_eval_dataloader()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()

        self.model.eval()

        for data in eval_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = self.model(inputs)

            metrics = compute_metrics(EvalPrediction(predictions=preds, labels=labels), scale=scale)

            epoch_psnr.update(metrics['psnr'], len(inputs))
            epoch_ssim.update(metrics['ssim'], len(inputs))

        print(f'scale:{str(scale)}      eval psnr: {epoch_psnr.avg:.2f}     ssim: {epoch_ssim.avg:.4f}')

        if epoch_psnr.avg > self.best_metric:
            self.best_epoch = epoch
            self.best_metric = epoch_psnr.avg

            print(f'best epoch: {epoch}, psnr: {epoch_psnr.avg:.6f}, ssim: {epoch_ssim.avg:.6f}')
            self.save_model()

    def _load_state_dict_in_model(self, state_dict):
        load_result = self.model.load_state_dict(state_dict, strict=False)

    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self.args.output_dir
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir)

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.
        Will only save from the main process.
        """

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not isinstance(self.model, PreTrainedModel):
            # Setup scale
            scale = self.model.config.scale
            if scale is not None:
                weights_name = WEIGHTS_NAME_SCALE.format(scale=scale)
            else:
                weights_name = WEIGHTS_NAME

            weights = copy.deepcopy(self.model.state_dict())
            torch.save(weights, os.path.join(output_dir, weights_name))
        else:
            self.model.save_pretrained(output_dir)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        """

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset

        return DataLoader(
            dataset=train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.
        """

        eval_dataset = self.eval_dataset
        if eval_dataset is None:
            eval_dataset = self.train_dataset

        return DataLoader(
            dataset=eval_dataset,
            batch_size=1,
        )
