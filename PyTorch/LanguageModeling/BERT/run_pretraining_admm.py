# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import shutil
import time
import logging
import argparse
import random
import os
from itertools import count
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, Dataset
from tensorboardX import SummaryWriter

from apex import amp

from modeling import BertForPreTraining, BertConfig
from optimization import BertLAMB

# from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from utils import is_main_process
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel.distributed import flat_dist_call
import amp_C
import apex_C
from apex.amp import _amp_state

from concurrent.futures import ProcessPoolExecutor

import sys
sys.path.append('/home/CORP.PKUSC.ORG/hatsu3/research/lab_projects/bert/notebooks/Cifar10_ADMM_Pruning_PyTorch')
from admm_manager_v2 import ProximalADMMPruningManager, PruningPhase, admm, test_irregular_sparsity
from args_to_yaml import *


global_step = 0
average_loss = 0.0
training_steps = 0
overflow_buf = None
most_recent_ckpts_paths = None
device = None
files = None
args = None


def plot_grad_flow(named_parameters, to_file=None):
    plt.clf()
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    if to_file: plt.savefig(to_file, dpi=200)
    else: return plt.gcf()


class LoggingMixin:
    def _log_scalar(self, *args, **kwargs):
        if not is_main_process(): return
        self.writer.add_scalar(*args, **kwargs)

    def _log_histogram(self, *args, **kwargs):
        if not is_main_process(): return
        self.writer.add_histogram(*args, **kwargs)

    def _log_figure(self, *args, **kwargs):
        if not is_main_process(): return
        self.writer.add_figure(*args, **kwargs)

    def _calc_avg_grads(self):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else: model = self.model
        avg_grads = []
        for n, p in model.named_parameters():
            if p.requires_grad and 'bias' not in n:
                avg_grads.append(p.grad.abs().mean().item())
        return np.array(avg_grads)

    def _plot_avg_grads(self):
        return plot_grad_flow(self.model.named_parameters())

    def _plot_avg_output(self):
        # print(self.forward_magnitudes)
        pd.Series(self.forward_magnitudes).plot()
        return plt.gcf()


class CheckpointMixin:
    @property
    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'admm': self.admm.state_dict(),
            'config': self.to_dict(),
            'optimizer': self.optimizer.state_dict(),
            'amp': amp.state_dict(),
            'timer': {
                'cur_phase': self.cur_phase.name,
                'cur_rho': self.cur_rho,
                'cur_step': self.cur_step,
            },
        }

    def resume_from(self, ckpt_path):
        self._load_full_checkpoint(ckpt_path)

    def _check_ckpt_config(self, state_dict):
        for k, v in state_dict.items():
            if getattr(self, k) == v: continue
            raise Exception(f'Config mismatch: ckpt: {k}={v} ~ now: {k}={getattr(self, k)}')

    def _load_full_checkpoint(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
        self._check_ckpt_config(state_dict['config'])
        self._load_model(state_dict['model'])
        self.admm.load_state_dict(state_dict['admm'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        amp.load_state_dict(state_dict['amp'])
        self._set_timer_status(state_dict['timer'])


class TimerMixin:
    def _timer_step(self):
        phase, rho, step = next(self._timer_iter)
        self.cur_phase = phase
        self.cur_rho = rho
        self.cur_step = step

    def _timer_gen(self):
        phase = PruningPhase.admm
        for rho_idx in range(self.rho_num):
            rho = self._calc_current_rho(rho_idx)
            yield from ((phase, rho, step) for step in range(self.admm_steps))
        phase = PruningPhase.masked_retrain
        yield from ((phase, None, step) for step in range(self.retrain_steps))
        for step in count():
            print(f'[WARNING] continue retraining after {self.retrain_steps} steps')
            print(f'Current step: {self.retrain_steps}+{step}. Please check your configuration.')

    def _set_timer_status(self, state_dict):
        if state_dict['cur_phase'] == PruningPhase.admm.name:
            self.prune()
        elif state_dict['cur_phase'] == PruningPhase.masked_retrain.name:
            self.retrain()
        else:
            raise ValueError(f"Invalid cur_phase: {state_dict['cur_phase']}")

        if self.cur_phase == PruningPhase.admm:
            possible_rhos = [self._calc_current_rho(i) for i in range(self.rho_num)]
            def check_rho(rho, eps=1e-6): return any(abs(rho-r) < eps for r in possible_rhos)
            assert check_rho(state_dict['cur_rho']), \
                f"Invalid cur_rho: {state_dict['cur_rho']}, should be in {possible_rhos}"
        else:
            assert state_dict['cur_rho'] is None, \
                f"Invalid cur_rho: {state_dict['cur_rho']}, should be None"
        self.cur_rho = state_dict['cur_rho']

        if self.cur_phase == PruningPhase.admm:
            assert 0 <= state_dict['cur_step'] < self.admm_steps, \
                f"Invalid cur_step: {state_dict['cur_step']}, should not be in [0, {self.admm_steps})"
        else:
            assert 0 <= state_dict['cur_step'] < self.retrain_steps, \
                f"Invalid cur_step: {state_dict['cur_step']}, should not be in [0, {self.retrain_steps})"
        self.cur_step = state_dict['cur_step']


class DebugMixin:
    def _setup_forward_hooks(self):
        self.forward_magnitudes = {}
        self.forward_handles = []
        for name, module in self.model.named_modules():
            def hook(module, inputs, outputs, name=name):
                # if name == "bert.encoder.layer.4.attention.self.softmax":
                #     print("bert.encoder.layer.4.attention.self.softmax", len(inputs),
                #           inputs[0].min().item(), inputs[0].max().item(),
                #           inputs[0].abs().mean().item())
                try:
                    if is_main_process():
                        pass
                        # try:
                        #     print(name, len(inputs),
                        #           inputs[0].min().item(), inputs[0].max().item(),
                        #           inputs[0].abs().mean().item(), end=' ')
                        # except: print(name, end=' ')
                        # print('->', outputs[0].min().item(), outputs[0].max().item(),
                        #       outputs[0].abs().mean().item())
                except:
                    print('***error:', name)
                # try:
                #     if isinstance(inputs, (list, tuple)): inputs = inputs[0]
                #     self.forward_magnitudes[name] = inputs.abs().mean().item()
                # except: pass
                # if isinstance(outputs, (list, tuple)): outputs = outputs[0]
                # self.forward_magnitudes[name] = outputs.abs().mean().item()
            self.forward_handles.append(module.register_forward_hook(hook))

    def _remove_forward_hooks(self):
        for handle in self.forward_handles:
            handle.remove()


class ProximalBertPruningManager(LoggingMixin, CheckpointMixin, TimerMixin, DebugMixin,
                                 ProximalADMMPruningManager):
    ATTRIBUTES = {
        'sparsity_config': None,
        'init_weights_path': None,

        'initial_rho': 0.0001,
        'rho_num': 1,
        'initial_lambda': 1,
        'cross_x': 1,
        'cross_f': 1,
        'update_freq': 300,  # steps
        'lr': None,
        'admm_steps': 3000,
        'retrain_steps': 3000,

        'sparsity_type': 'threshold',
        'arch': 'bert_base_uncased',
        'optimizer_type': 'NVLAMB',
        'dataset_type': 'EnwikiBookcorpus',
        'save_dir': None,
        'overwrite': False,
        'admm_ckpt_steps': 500,
        'retrain_ckpt_steps': 500,

        'fp16': True,
        'tensorboard_logdir': '.',
        'tensorboard_json_path': './all_scalars.json',
    }

    # noinspection PyMethodOverriding
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cur_rho = None
        self.cur_step = None
        self.writer = None
        self._timer_iter = self._timer_gen()
        self.sparsity_tester = test_irregular_sparsity

    def __del__(self):
        if self.writer:
            self.writer.export_scalars_to_json(self.tensorboard_json_path)
            self.writer.close()
        if getattr(self, 'forward_handles', None):
            self._remove_forward_hooks()

    # noinspection PyMethodOverriding
    def setup_learner(self, model, optimizer, train_loader):
        if is_main_process():
            if Path(self.tensorboard_logdir).is_dir() and self.overwrite:
                shutil.rmtree(self.tensorboard_logdir)
            # Path(self.tensorboard_logdir).mkdir(exist_ok=False, parents=True)
            self.writer = SummaryWriter(logdir=self.tensorboard_logdir, flush_secs=60)
        torch.distributed.barrier()

        self.update_freq *= args.gradient_accumulation_steps
        self.admm_steps *= args.gradient_accumulation_steps
        self.retrain_steps *= args.gradient_accumulation_steps
        self.admm_ckpt_steps *= args.gradient_accumulation_steps
        self.retrain_ckpt_steps *= args.gradient_accumulation_steps

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader

        if is_main_process():
            self._setup_forward_hooks()  # debugging

    def update_lr(self, step):
        if self.cur_phase == PruningPhase.admm:
            new_lr = admm.admm_adjust_learning_rate_per_step(
                step, args=argparse.Namespace(
                    lr=self.lr, admm_update_freq=self.update_freq,
                )
            )
            self.optimizer.set_lr(new_lr)
        else:
            new_lr = self.lr * ((1 - step / self.retrain_steps) ** 0.5)
            self.optimizer.set_lr(new_lr)

    # noinspection PyMethodOverriding
    def append_admm_loss(self, loss, step, return_losses=False):
        assert self.cur_phase == PruningPhase.admm
        args = argparse.Namespace(
            admm=True, verbose=True,
            admm_update_freq=self.update_freq,
            lamda=self.initial_lambda,
            sparsity_type=self.sparsity_type,
            cross_x=self.cross_x,
            cross_f=self.cross_f,
        )
        writer = self.writer if is_main_process() else None
        admm.proximal_update_per_step(args, self.admm, self.model, step, writer=writer)
        admm.admm_update_per_step(args, self.admm, self.model, step)
        losses = admm.append_admm_loss(args, self.admm, self.model, loss)
        loss, admm_loss, mixed_loss = losses
        if not return_losses: return mixed_loss
        else: return losses

    def admm_prune(self):
        resumed = self.cur_phase is not None
        if self.cur_phase == PruningPhase.masked_retrain:
            print('ADMM pruning has finished. Skipping admm_prune...')
            return
        if self.cur_phase == PruningPhase.admm:
            init_rho_idx = [self._calc_current_rho(i) for i in range(self.rho_num)].index(self.cur_rho)
        else: init_rho_idx = 0
        # self.prune()
        for i in range(init_rho_idx, self.rho_num):
            if not resumed or i > init_rho_idx:  # do not skip initialization in next iteration
                self._load_ckpt_admm_prune(i)
                current_rho = self._calc_current_rho(i)
                current_lamda = self._calc_current_lamda(i)
                self._init_admm(rho=current_rho, lamda=current_lamda)
                admm.admm_initialization(argparse.Namespace(
                    admm=(self.cur_phase == PruningPhase.admm),
                    sparsity_type=self.sparsity_type,
                ), self.admm, self.model)  # initialize Z variable
            else: current_rho = self.cur_rho
            self._train_admm_prune(current_rho)

    def masked_retrain(self):
        resumed = self.cur_phase == PruningPhase.masked_retrain
        if not resumed:
            # self.retrain()
            self._load_ckpt_masked_retrain()
            self._init_admm(rho=self.initial_rho, lamda=self.initial_lambda)
            args = argparse.Namespace(sparsity_type=self.sparsity_type)
            if isinstance(self.model, nn.DataParallel):
                model = self.model.module
            else: model = self.model
            admm.hard_prune(args, self.admm, model)
        if is_main_process():
            self.sparsity_tester(self.model)
        self._train_masked_retrain()
        if is_main_process():
            self.sparsity_tester(self.model)

    def _init_admm(self, rho, lamda):
        self.admm = admm.ADMM(self.model, file_name=self.sparsity_config, rho=rho, lamda=lamda)

    def _train_masked_retrain(self):
        resumed = self.cur_phase == PruningPhase.masked_retrain
        init_step = 0 if not resumed else self.cur_step
        self.setup_masking_hooks()
        for step in range(init_step, self.retrain_steps):
            self._timer_step()
            self._train_one_step(step)
            if step % self.retrain_ckpt_steps == 0 and is_main_process():
                self.save_checkpoint_retrain(step=step)
        if is_main_process():
            self.save_checkpoint_retrain(step=self.retrain_steps)
        torch.distributed.barrier()
        self.remove_masking_hooks()

    def _train_admm_prune(self, current_rho):
        resumed = self.cur_rho == current_rho
        init_step = 0 if not resumed else self.cur_step
        for step in range(init_step, self.admm_steps):
            self._timer_step()
            self._train_one_step(step)
            if step % self.admm_ckpt_steps == 0 and is_main_process():
                self.save_checkpoint_prune(rho=current_rho, step=step)
        if is_main_process():
            self.save_checkpoint_prune(rho=current_rho)
        torch.distributed.barrier()

    def _train_one_step(self, my_step):
        global args, average_loss, global_step, training_steps, device, overflow_buf, most_recent_ckpts_paths, files
        epoch, f_id, step, batch = next(self.train_loader)

        batch = [t.to(device) for t in batch]
        input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
        loss, masked_lm_acc, next_sentence_acc = self.model(
            input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
            masked_lm_labels=masked_lm_labels, next_sentence_label=next_sentence_labels,
            checkpoint_activations=args.checkpoint_activations, return_acc=True)
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.

        if self.cur_phase == PruningPhase.admm:
            self._log_scalar(f'loss/admm_orig_loss_rho{self.cur_rho}', loss.item(), global_step=my_step)
            self._log_scalar(f'acc/admm_masked_lm_acc_rho{self.cur_rho}', masked_lm_acc.item(), global_step=my_step)
            self._log_scalar(f'acc/admm_next_sentence_acc_rho{self.cur_rho}', next_sentence_acc.item(), global_step=my_step)
        else:
            self._log_scalar('loss/retrain_loss', loss.item(), global_step=my_step)
            self._log_scalar('acc/retrain_masked_lm_acc', masked_lm_acc.item(), global_step=my_step)
            self._log_scalar('acc/retrain_next_sentence_acc', next_sentence_acc.item(), global_step=my_step)

        if self.cur_phase == PruningPhase.admm:
            orig_loss, admm_loss, loss = self.append_admm_loss(loss, my_step, return_losses=True)
            # print("***", [loss.item() for loss in admm_loss.values()])  # [nan, nan, nan...]
            # print("***", sum(admm_loss.values()).item())  # nan
            self._log_scalar(f'loss/admm_admm_loss_rho{self.cur_rho}', sum(admm_loss.values()).item(), global_step=my_step)
            self._log_scalar(f'loss/admm_mixed_loss_rho{self.cur_rho}', loss.item(), global_step=my_step)

        divisor = args.gradient_accumulation_steps
        if args.gradient_accumulation_steps > 1:
            if not args.allreduce_post_accumulation:
                # this division was merged into predivision
                loss = loss / args.gradient_accumulation_steps
                divisor = 1.0
        if args.fp16:
            with amp.scale_loss(loss, self.optimizer, delay_overflow_check=args.allreduce_post_accumulation) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        average_loss += loss.item()

        # if my_step % 500 == 0:
        #     self._log_figure('output/barplot', self._plot_avg_output(), global_step=my_step)
        #
        # if my_step % 2000 == 0:
        #     # plot_grad_flow(self.model.named_parameters(), f'grad_step{my_step}.png')
        #     print(self._calc_avg_grads())
        #     self._log_histogram('gradient/grad_hist', self._calc_avg_grads(), global_step=my_step)
        #     self._log_figure('gradient/grad_barplot', self._plot_avg_grads(), global_step=my_step)

        self.update_lr(my_step)
        cur_lr = self.optimizer.manual_lr
        if self.cur_phase == PruningPhase.admm:
            self._log_scalar(f'lr/admm_lr_rho{self.cur_rho}', cur_lr, global_step=my_step)
        else:
            self._log_scalar('lr/retrain_lr', cur_lr, global_step=my_step)

        if training_steps % args.gradient_accumulation_steps == 0:
            global_step = take_optimizer_step(args, self.optimizer, self.model, overflow_buf, global_step)

        # if global_step >= args.max_steps:
        #     print_final_loss(training_steps, average_loss, divisor)

        if training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
            if is_main_process():
                print("Step:{} Average Loss = {} Step Loss = {} LR {}".format(
                    global_step, average_loss / (args.log_freq * divisor),
                    loss.item() * args.gradient_accumulation_steps / divisor,
                    self.optimizer.manual_lr))  # self.optimizer.param_groups[0]['lr']))
            average_loss = 0

        # if global_step >= args.max_steps or training_steps % (
        #         args.num_steps_per_checkpoint * args.gradient_accumulation_steps) == 0:
        #     save_checkpoint(self.model, self.optimizer, global_step, files, f_id, most_recent_ckpts_paths)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# noinspection PyUnusedLocal
def create_pretraining_dataset(input_file, max_pred_length, shared_list, args):
    train_data = pretraining_dataset(input_file=input_file, max_pred_length=max_pred_length)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size * args.n_gpu, num_workers=4,
                                  pin_memory=True)
    return train_dataloader, input_file


class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input_[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input_[index].astype(np.int64))) for indice, input_ in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]


def parse_arguments():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")

    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")

    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=1.0,
                        help='frequency of logging loss.')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint.")
    parser.add_argument('--resume_step',
                        type=int,
                        default=-1,
                        help="Step to resume training from.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=100,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Whether to train with seq len 512")
    parser.add_argument('--allreduce_post_accumulation',
                        default=False,
                        action='store_true',
                        help="Whether to do allreduces during gradient accumulation steps.")
    parser.add_argument('--allreduce_post_accumulation_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to do fp16 allreduce post accumulation.")
    parser.add_argument('--accumulate_into_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use fp16 gradient accumulators.")
    parser.add_argument('--phase1_end_step',
                        type=int,
                        default=7038,
                        help="Number of training steps in Phase1 - seq len 128")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--admm_resume_from_checkpoint",
                        default=None,
                        type=str,
                        help="Resume training from checkpoint.")

    # parser.add_argument("--admm_config",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The ADMM Pruning config")

    args = parser.parse_args()
    return args


def setup_training(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert (torch.cuda.is_available())

    if args.local_rank == -1:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("device %s n_gpu %d distributed training %r", device, args.n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train:
        raise ValueError(" `do_train`  must be True.")

    if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and (
            os.listdir(args.output_dir) and any([i.startswith('ckpt') for i in os.listdir(args.output_dir)])):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if not args.resume_from_checkpoint:
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args


# noinspection PyUnresolvedReferences
def prepare_model_and_optimizer(args, device):
    # Prepare model
    config = BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    # if config.vocab_size % 8 != 0:
    #     config.vocab_size += 8 - (config.vocab_size % 8)
    model = BertForPreTraining(config)

    checkpoint = None
    if not args.resume_from_checkpoint:
        global_step = 0
    else:
        if args.resume_step == -1:
            model_names = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
            args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])
        global_step = args.resume_step

        checkpoint = torch.load(os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step)), map_location="cpu")
        model.load_state_dict(checkpoint['model'], strict=False)
        if args.phase2:
            global_step -= args.phase1_end_step
        if is_main_process():
            print("resume step from ", args.resume_step)

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = []
    names = []

    count = 1
    for n, p in param_optimizer:
        count += 1
        if not any(nd in n for nd in no_decay):
            optimizer_grouped_parameters.append({'params': [p], 'weight_decay': 0.01, 'name': n})
            names.append({'params': [n], 'weight_decay': 0.01})
        if any(nd in n for nd in no_decay):
            optimizer_grouped_parameters.append({'params': [p], 'weight_decay': 0.00, 'name': n})
            names.append({'params': [n], 'weight_decay': 0.00})

    optimizer = BertLAMB(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=args.max_steps)
    if args.fp16:
        opt_level = "O2"  # originally O2
        if args.loss_scale == 0:
            # optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            if opt_level == "O1": master_weights = None
            else: master_weights = False if args.accumulate_into_fp16 else True
            model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level, loss_scale="dynamic",
                                              master_weights=master_weights)
        else:
            # optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level, loss_scale=args.loss_scale,
                                              master_weights=False if args.accumulate_into_fp16 else True)
        amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    if args.resume_from_checkpoint:
        if args.phase2:
            keys = list(checkpoint['optimizer']['state'].keys())
            # Override hyperparameters from Phase 1
            for key in keys:
                checkpoint['optimizer']['state'][key]['step'] = global_step
            for iter_, item in enumerate(checkpoint['optimizer']['param_groups']):
                checkpoint['optimizer']['param_groups'][iter_]['t_total'] = args.max_steps
                checkpoint['optimizer']['param_groups'][iter_]['warmup'] = args.warmup_proportion
                checkpoint['optimizer']['param_groups'][iter_]['lr'] = args.learning_rate
        optimizer.load_state_dict(checkpoint['optimizer'])  # , strict=False)

        # Restore AMP master parameters
        if args.fp16:
            optimizer._lazy_init_maybe_master_weights()
            optimizer._amp_stash.lazy_init_called = True
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param, saved_param in zip(amp.master_params(optimizer), checkpoint['master params']):
                param.data.copy_(saved_param.data)

    if args.local_rank != -1:
        if not args.allreduce_post_accumulation:
            model = DDP(model, message_size=250000000, gradient_predivide_factor=torch.distributed.get_world_size())
        else:
            flat_dist_call([param.data for param in model.parameters()], torch.distributed.broadcast, (0,))
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, optimizer, checkpoint, global_step


def take_optimizer_step(args, optimizer, model, overflow_buf, global_step):
    if args.allreduce_post_accumulation:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        scaler = _amp_state.loss_scalers[0]
        master_grads = [p.grad for p in amp.master_params(optimizer) if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        allreduce_dtype = torch.float16 if args.allreduce_post_accumulation_fp16 else torch.float32
        flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(65536,
                                 overflow_buf,
                                 [master_grads, allreduced_views],
                                 scaler.loss_scale() / (
                                             torch.distributed.get_world_size() * args.gradient_accumulation_steps))
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        torch.distributed.all_reduce(flat_raw)
        # 4. combine unscaling and unflattening of allreduced gradient
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(65536,
                                 overflow_buf,
                                 [allreduced_views, master_grads],
                                 1. / scaler.loss_scale())
        # 5. update loss scale
        scaler = _amp_state.loss_scalers[0]
        old_overflow_buf = scaler._overflow_buf
        scaler._overflow_buf = overflow_buf
        had_overflow = scaler.update_scale()
        scaler._overfloat_buf = old_overflow_buf
        # 6. call optimizer step function
        if had_overflow == 0:
            optimizer.step()
            global_step += 1
        else:
            # Overflow detected, print message and clear gradients
            if is_main_process():
                print(("Rank {} :: Gradient overflow.  Skipping step, " +
                       "reducing loss scale to {}").format(
                    torch.distributed.get_rank(),
                    scaler.loss_scale()))
            if _amp_state.opt_properties.master_weights:
                for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
        for param in model.parameters():
            param.grad = None
    else:
        optimizer.step()
        # optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None
        global_step += 1

    return global_step


def save_checkpoint(model, optimizer, global_step, files, f_id, most_recent_ckpts_paths):
    if is_main_process():
        # Save a trained model
        logger.info("** ** * Saving fine - tuned model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        if args.resume_step < 0 or not args.phase2:
            output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step))
        else:
            output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step + args.phase1_end_step))
        if args.do_train:
            torch.save({'model': model_to_save.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'master params': list(amp.master_params(optimizer)),
                        'files': [f_id] + files}, output_save_file)

            most_recent_ckpts_paths.append(output_save_file)
            if len(most_recent_ckpts_paths) > 3:
                ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                os.remove(ckpt_to_be_removed)


def print_final_loss(training_steps, average_loss, divisor):
    last_num_steps = int(training_steps / args.gradient_accumulation_steps) % args.log_freq
    last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
    average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()
    average_loss = average_loss / (last_num_steps * divisor)
    if torch.distributed.is_initialized():
        average_loss /= torch.distributed.get_world_size()
        torch.distributed.all_reduce(average_loss)
    if is_main_process():
        logger.info("Total Steps:{} Final Loss = {}".format(
            training_steps / args.gradient_accumulation_steps, average_loss.item()))


def train_one_file(model, optimizer, train_dataloader, f_id):
    global training_steps, global_step, average_loss, overflow_buf, most_recent_ckpts_paths, files, device

    train_iter = tqdm(train_dataloader, desc="Iteration") if is_main_process() else train_dataloader
    for step, batch in enumerate(train_iter):

        training_steps += 1
        batch = [t.to(device) for t in batch]
        input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
        loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                     masked_lm_labels=masked_lm_labels, next_sentence_label=next_sentence_labels,
                     checkpoint_activations=args.checkpoint_activations)
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.

        divisor = args.gradient_accumulation_steps
        if args.gradient_accumulation_steps > 1:
            if not args.allreduce_post_accumulation:
                # this division was merged into predivision
                loss = loss / args.gradient_accumulation_steps
                divisor = 1.0
        if args.fp16:
            with amp.scale_loss(loss, optimizer,
                                delay_overflow_check=args.allreduce_post_accumulation) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        average_loss += loss.item()

        if training_steps % args.gradient_accumulation_steps == 0:
            global_step = take_optimizer_step(args, optimizer, model, overflow_buf, global_step)

        if global_step >= args.max_steps:
            print_final_loss(training_steps, average_loss, divisor)

        elif training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
            if is_main_process():
                print("Step:{} Average Loss = {} Step Loss = {} LR {}".format(
                    global_step, average_loss / (args.log_freq * divisor),
                    loss.item() * args.gradient_accumulation_steps / divisor,
                    optimizer.param_groups[0]['lr']))
            average_loss = 0

        if global_step >= args.max_steps or training_steps % (
                args.num_steps_per_checkpoint * args.gradient_accumulation_steps) == 0:
            save_checkpoint(model, optimizer, global_step, files, f_id, most_recent_ckpts_paths)


def setup_files(epoch, checkpoint):
    global args
    if not args.resume_from_checkpoint or epoch > 0 or args.phase2:
        files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                 os.path.isfile(os.path.join(args.input_dir, f)) and 'training' in f]
        files.sort()
        num_files = len(files)
        random.shuffle(files)
        f_start_id = 0
    else:
        f_start_id = checkpoint['files'][0]
        files = checkpoint['files'][1:]
        args.resume_from_checkpoint = False
        num_files = len(files)

    shared_file_list = {}

    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > num_files:
        remainder = torch.distributed.get_world_size() % num_files
        data_file = files[(f_start_id * torch.distributed.get_world_size() + torch.distributed.get_rank() + remainder * f_start_id) % num_files]
    else:
        remainder = None
        data_file = files[(f_start_id * torch.distributed.get_world_size() + torch.distributed.get_rank()) % num_files]

    previous_file = data_file
    return f_start_id, files, num_files, data_file, previous_file, shared_file_list, remainder


def prefetch_next_file(pool, f_id, files, num_files, previous_file, shared_file_list, remainder):
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if torch.distributed.get_world_size() > num_files:
        data_file = files[(f_id * world_size + rank + remainder * f_id) % num_files]
    else:
        data_file = files[(f_id * world_size + rank) % num_files]

    logger.info("file no %s file %s" % (f_id, previous_file))
    previous_file = data_file
    dataset_future = pool.submit(
        create_pretraining_dataset, data_file,
        args.max_predictions_per_seq, shared_file_list, args
    )
    return previous_file, dataset_future


def infinite_data_loader(checkpoint):
    global global_step, training_steps, overflow_buf, files
    epoch = 0
    pool = ProcessPoolExecutor(1)

    # Note: We loop infinitely over epochs, termination is handled via iteration count
    while True:  # one epoch
        (f_start_id, files, num_files, data_file, previous_file,
         shared_file_list, remainder) = setup_files(epoch, checkpoint)

        """ first data_loader """
        train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler,
            batch_size=args.train_batch_size * args.n_gpu,
            num_workers=4, pin_memory=True,
        )
        # shared_file_list["0"] = (train_dataloader, data_file)

        overflow_buf = None
        if args.allreduce_post_accumulation:
            overflow_buf = torch.cuda.IntTensor([0])

        for f_id in range(f_start_id + 1, len(files)):
            previous_file, dataset_future = prefetch_next_file(
                pool, f_id, files, num_files, previous_file, shared_file_list, remainder)

            train_iter = tqdm(train_dataloader, desc="Iteration") if is_main_process() else train_dataloader
            for step, batch in enumerate(train_iter):
                training_steps += 1
                yield epoch, f_id, step, batch

            # if global_step >= args.max_steps:
            #     del train_dataloader
            #     # thread.join()
            #     return args  # raise StopIteration

            del train_dataloader
            # thread.join()
            # Make sure pool has finished and switch train_dataloader
            # NOTE: Will block until complete
            train_dataloader, data_file = dataset_future.result(timeout=None)

        epoch += 1


def parse_my_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_yaml', type=str, help="yaml configuration file for pretraining")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_known_args()[0]
    return args.config_yaml, args.local_rank


def main():
    global global_step, average_loss, training_steps, most_recent_ckpts_paths, device, args
    config_yaml, local_rank = parse_my_arguments()
    args = args_from_yaml(config_yaml)
    args.local_rank = local_rank
    # args = parse_arguments()
    device, args = setup_training(args)

    model, optimizer, checkpoint, global_step = prepare_model_and_optimizer(args, device)

    if is_main_process(): print("SEED {}".format(args.seed))

    if args.do_train:
        if is_main_process():
            logger.info("***** Running training *****")
            # logger.info("  Num examples = %d", len(train_data))
            logger.info("  Batch size = %d", args.train_batch_size)
            print("  LR = ", args.learning_rate)
            print("Training. . .")

        model.train()
        most_recent_ckpts_paths = []
        average_loss = 0.0  # averaged loss every args.log_freq steps
        training_steps = 0

        train_loader = infinite_data_loader(checkpoint)
        # prune_manager = ProximalBertPruningManager.from_yaml_file(args.admm_config)
        prune_manager = ProximalBertPruningManager.from_dict(vars(args))
        if is_main_process(): print(prune_manager.to_json_string())
        prune_manager.setup_learner(model, optimizer, train_loader)
        if args.admm_resume_from_checkpoint:
            prune_manager.resume_from(args.admm_resume_from_checkpoint)
        prune_manager.admm_prune()
        prune_manager.masked_retrain()


if __name__ == "__main__":
    now = time.time()
    main()
    if is_main_process():
        print("Total time taken {}".format(time.time() - now))
