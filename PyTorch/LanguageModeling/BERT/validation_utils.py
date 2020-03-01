"""
Evaluate model on SQuAD v1.1 and GLUE
Usage:
    # First, check out if default values in DEFAULT_GLUE_ARGS are suitable for your use
    acc = validate_on_glue(
        'my_checkpoint.pth.tar', stage=PruningPhase.masked_retrain,     # current stage
        task_name='MRPC', data_dir='data/glue/MRPC',                    # changes between tasks
        seed=114514,            # remember to change seeds between runs
        train_batch_size=16,    # **kwargs: other parameters you want to change (see DEFAULT_GLUE_ARGS)
    )
"""

import argparse
import json
import os
import pickle
import random
import re
import string
import sys
from collections import Counter

import numpy as np
import torch
import yaml
from apex import amp
from torch import nn

import run_glue
import run_squad
from admm_manager_v2 import PruningPhase
from args_to_yaml import yaml_ordered_load
from run_glue_admm import get_parameter_by_name


# TODO: load checkpoint on an idle GPU to avoid OOM error


""" >>> Remember to set default value for arguments that do not change between runs <<< """


DEFAULT_GLUE_ARGS = {
    'init_checkpoint': '',
    'sparsity_config': 'bert_base_sparsity_config.example.yaml',

    'task_name': 'MRPC',
    'data_dir': 'data/glue/MRPC',

    'bert_model': 'bert-base-uncased',
    'do_lower_case': True,
    'max_seq_length': 128,
    'cache_dir': '',

    'learning_rate': 5.0e-06,
    'train_batch_size': 12,
    'eval_batch_size': 8,
    'gradient_accumulation_steps': 1,

    'max_steps': -1.0,
    'num_train_epochs': 2.0,
    'warmup_proportion': 0.1,

    'fp16': True,
    'loss_scale': 0,
    'seed': 42,
}

DEFAULT_SQUAD_ARGS = {
    'init_checkpoint': '',
    'sparsity_config': 'bert_base_sparsity_config.example.yaml',

    'version_2_with_negative': False,
    'train_file': 'data/download/squad/v1.1/train-v1.1.json',
    'predict_file': 'data/download/squad/v1.1/dev-v1.1.json',
    'max_answer_length': 30,
    'max_query_length': 64,
    'doc_stride': 128,

    'bert_model': 'bert-base-uncased',
    'do_lower_case': True,
    'max_seq_length': 384,
    'config_file': 'data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_config.json',
    'vocab_file': 'data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt',

    'learning_rate': 3.0e-05,
    'train_batch_size': 3,
    'predict_batch_size': 3,
    'gradient_accumulation_steps': 1,

    'max_steps': -1.0,
    'num_train_epochs': 2.0,
    'warmup_proportion': 0.1,

    'old': False,  # Use Old FP16 Optimizer
    'fp16': True,
    'loss_scale': 0,
    'seed': 1,
    'log_freq': 50,
    'verbose_logging': False,
    'null_score_diff_threshold': 0.0,
}


""" Utils """


def _parse_args(default_args, **kwargs):
    args_dict = default_args.copy()
    args_dict.update(kwargs)
    return argparse.Namespace(**args_dict)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_checkpoint(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    state_dict = state_dict.get('model', state_dict)
    model.load_state_dict(state_dict, strict=False)


def _hard_mask(model, sparsity_config, threshold=1e-4):
    print('Hard Prune')
    prune_ratios = yaml_ordered_load(sparsity_config)['prune_ratios']
    for name, weight in model.named_parameters():
        if name.startswith('module.'):
            name = name[len('module.'):]
        if name not in prune_ratios: continue
        weight_np = weight.cpu().detach().numpy()
        weight_np[np.abs(weight_np) < threshold] = 0
        weight.data = torch.from_numpy(weight_np).cuda()


""" GLUE Validation """


def validate_on_glue(checkpoint_path, stage: PruningPhase, task_name, data_dir, seed, **kwargs):
    args = _parse_args(DEFAULT_GLUE_ARGS, init_checkpoint=checkpoint_path,
                       task_name=task_name, data_dir=data_dir, seed=seed, **kwargs)
    model, processor, tokenizer, label_list = _train_glue(args, stage)
    eval_loss, eval_accuracy = _validate_glue(args, model, processor, tokenizer, label_list)
    return eval_accuracy


def _train_glue(args, stage):
    _set_seed(args.seed)

    task_name = args.task_name.lower()
    processor = {
        "cola": run_glue.ColaProcessor,
        "mnli": run_glue.MnliProcessor,
        "mrpc": run_glue.MrpcProcessor,
    }[task_name]()
    label_list = processor.get_labels()
    num_labels = {"cola": 2, "mnli": 3, "mrpc": 2, }[task_name]
    tokenizer = run_glue.BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    cache_dir = os.path.join(run_glue.PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_-1')

    model = run_glue.BertForSequenceClassification.from_pretrained(
        args.bert_model, cache_dir=cache_dir, num_labels=num_labels)
    _load_checkpoint(model, args.init_checkpoint)

    if stage == PruningPhase.admm:
        _hard_mask(model, args.sparsity_config)
    if args.fp16: model = model.half()
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    plain_model = getattr(model, 'module', model)

    with open(args.sparsity_config, 'r') as f:
        raw_dict = yaml.load(f, Loader=yaml.SafeLoader)
        masks = dict.fromkeys(raw_dict['prune_ratios'].keys())
        for param_name in list(masks.keys()):
            if get_parameter_by_name(plain_model, param_name) is None:
                print(f'[WARNING] Cannot find {param_name}')
                del masks[param_name]

    for param_name in masks:
        param = get_parameter_by_name(plain_model, param_name)
        non_zero_mask = torch.ne(param, 0).to(param.dtype)
        masks[param_name] = non_zero_mask

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.fp16_utils.fp16_optimizer import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = run_glue.BertAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            warmup=args.warmup_proportion,
            t_total=num_train_optimization_steps,
        )

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    train_features = run_glue.convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer)
    run_glue.logger.info("***** Running training *****")
    run_glue.logger.info("  Num examples = %d", len(train_examples))
    run_glue.logger.info("  Batch size = %d", args.train_batch_size)
    run_glue.logger.info("  Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = run_glue.TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = run_glue.RandomSampler(train_data)
    train_dataloader = run_glue.DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    model.train()
    for _ in run_glue.trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(run_glue.tqdm(train_dataloader, desc="Iteration")):
            if args.max_steps > 0 and global_step > args.max_steps:
                break
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            if torch.cuda.device_count() > 1: loss = loss.mean()
            loss = loss / args.gradient_accumulation_steps

            if args.fp16: optimizer.backward(loss)
            else: loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * run_glue.warmup_linear(
                        global_step / num_train_optimization_steps, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                plain_model = getattr(model, 'module', model)
                for param_name, mask in masks.items():
                    get_parameter_by_name(plain_model, param_name).data *= mask

    return model, processor, tokenizer, label_list


def _validate_glue(args, model, processor, tokenizer, label_list):
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = run_glue.convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)
    run_glue.logger.info("***** Running evaluation *****")
    run_glue.logger.info("  Num examples = %d", len(eval_examples))
    run_glue.logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = run_glue.TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = run_glue.SequentialSampler(eval_data)
    eval_dataloader = run_glue.DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for input_ids, input_mask, segment_ids, label_ids in run_glue.tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        label_ids = label_ids.cuda()

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = run_glue.accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    return eval_loss, eval_accuracy


""" SQuAD Evaluation """


def validate_on_squad(checkpoint_path, stage: PruningPhase, seed, **kwargs):
    args = _parse_args(DEFAULT_SQUAD_ARGS, init_checkpoint=checkpoint_path, seed=seed, **kwargs)
    model, tokenizer = _train_squad(args, stage)
    result = _validate_squad(args, model, tokenizer)
    exact_match, f1 = result['exact_match'], result['f1']
    return exact_match, f1


def _train_squad(args, stage):
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    _set_seed(args.seed)

    tokenizer = run_squad.BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=512)  # for bert large
    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None

    train_examples = run_squad.read_squad_examples(
        input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    config = run_squad.BertConfig.from_json_file(args.config_file)
    model: nn.Module = run_squad.BertForQuestionAnswering(config)
    _load_checkpoint(model, args.init_checkpoint)

    if stage == PruningPhase.admm:
        _hard_mask(model, args.sparsity_config)

    model.cuda()
    if args.fp16 and args.old:
        model.half()

    with open(args.sparsity_config, 'r') as f:
        raw_dict = yaml.load(f, Loader=yaml.SafeLoader)
        masks = dict.fromkeys(raw_dict['prune_ratios'].keys())

    plain_model = getattr(model, 'module', model)

    for param_name in masks:
        param = get_parameter_by_name(plain_model, param_name)
        if param is None: raise Exception(f'Cannot find {param_name}')
        non_zero_mask = torch.ne(param, 0).to(param.dtype)
        masks[param_name] = non_zero_mask

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            # from fused_adam_local import FusedAdamBert as FusedAdam
            from apex.optimizers import FusedAdam
            from apex.fp16_utils.fp16_optimizer import FP16_Optimizer
            # from apex.contrib.optimizers import FP16_Optimizer
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        # import ipdb; ipdb.set_trace()
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)

        if args.loss_scale == 0:
            if args.old:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                  loss_scale="dynamic")
        else:
            if args.old:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            else:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                  loss_scale=args.loss_scale)
        if not args.old and args.do_train:
            scheduler = run_squad.LinearWarmUpScheduler(
                optimizer, warmup=args.warmup_proportion, total_steps=num_train_optimization_steps)

    else:
        optimizer = run_squad.BertAdam(
            optimizer_grouped_parameters, lr=args.learning_rate,
            warmup=args.warmup_proportion, t_total=num_train_optimization_steps)

    model = torch.nn.DataParallel(model)

    global_step = 0
    cached_train_features_file = args.train_file + '_{0}_{1}_{2}_{3}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride),
        str(args.max_query_length))
    # train_features = None
    try:
        with open(cached_train_features_file, "rb") as reader:
            train_features = pickle.load(reader)
    except:
        train_features = run_squad.convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True)
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            run_squad.logger.info("  Saving train features into cached file %s", cached_train_features_file)
            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)

    run_squad.logger.info("***** Running training *****")
    run_squad.logger.info("  Num orig examples = %d", len(train_examples))
    run_squad.logger.info("  Num split examples = %d", len(train_features))
    run_squad.logger.info("  Batch size = %d", args.train_batch_size)
    run_squad.logger.info("  Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    train_data = run_squad.TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions)
    train_sampler = run_squad.RandomSampler(train_data)
    train_dataloader = run_squad.DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    model.train()
    for _ in run_squad.trange(int(args.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(run_squad.tqdm(train_dataloader, desc="Iteration")):
            # Terminate early for benchmarking

            if args.max_steps > 0 and global_step > args.max_steps:
                break

            if torch.cuda.device_count() == 1:
                batch = tuple(t.cuda() for t in batch)  # multi-gpu does scattering it-self
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
            loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
            if torch.cuda.device_count() > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                if args.old:
                    # noinspection PyUnboundLocalVariable
                    optimizer.backward(loss)
                else:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
            else:
                loss.backward()
            # if args.fp16:
            #    optimizer.backward(loss)
            # else:
            #    loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up for BERT which FusedAdam doesn't do
                    if not args.old:
                        # noinspection PyUnboundLocalVariable
                        scheduler.step()
                    else:
                        lr_this_step = args.learning_rate * run_squad.warmup_linear(
                            global_step / num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                plain_model = getattr(model, 'module', model)
                for param_name, mask in masks.items():
                    param = get_parameter_by_name(plain_model, param_name)
                    param.data *= mask.to(param.dtype)

            if step % args.log_freq == 0:
                # logger.info("Step {}: Loss {}, LR {} ".format(global_step, loss.item(), lr_this_step))
                run_squad.logger.info(
                    "Step {}: Loss {}, LR {} ".format(global_step, loss.item(), optimizer.param_groups[0]['lr']))

    return model, tokenizer


def _calc_metric_squad(dataset_file, prediction_file):
    """ copied from squad/v1.1/evaluate-v1.1.py """

    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(prediction, ground_truth):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def exact_match_score(prediction, ground_truth):
        return normalize_answer(prediction) == normalize_answer(ground_truth)

    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    def evaluate(dataset, predictions):
        f1 = exact_match = total = 0
        for article in dataset:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    total += 1
                    if qa['id'] not in predictions:
                        message = 'Unanswered question ' + qa['id'] + \
                                  ' will receive score 0.'
                        print(message, file=sys.stderr)
                        continue
                    ground_truths = list(map(lambda x: x['text'], qa['answers']))
                    prediction = predictions[qa['id']]
                    exact_match += metric_max_over_ground_truths(
                        exact_match_score, prediction, ground_truths)
                    f1 += metric_max_over_ground_truths(
                        f1_score, prediction, ground_truths)

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total

        return {'exact_match': exact_match, 'f1': f1}

    with open(dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
    with open(prediction_file) as prediction_file:
        predictions = json.load(prediction_file)

    result = evaluate(dataset, predictions)
    print(json.dumps(result))
    return result


def _validate_squad(args, model, tokenizer):
    eval_examples = run_squad.read_squad_examples(
        input_file=args.predict_file,
        is_training=False,
        version_2_with_negative=args.version_2_with_negative)

    eval_features = run_squad.convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=False)

    run_squad.logger.info("***** Running predictions *****")
    run_squad.logger.info("  Num orig examples = %d", len(eval_examples))
    run_squad.logger.info("  Num split examples = %d", len(eval_features))
    run_squad.logger.info("  Batch size = %d", args.predict_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = run_squad.TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    # Run prediction for full data
    eval_sampler = run_squad.SequentialSampler(eval_data)
    eval_dataloader = run_squad.DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    model.eval()
    all_results = []
    run_squad.logger.info("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in run_squad.tqdm(eval_dataloader, desc="Evaluating"):
        if len(all_results) % 1000 == 0:
            run_squad.logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(run_squad.RawResult(
                unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))
    output_prediction_file = os.path.join("predictions.json")
    output_nbest_file = os.path.join("nbest_predictions.json")
    output_null_log_odds_file = os.path.join("null_odds.json")
    run_squad.write_predictions(
        eval_examples, eval_features, all_results,
        args.n_best_size, args.max_answer_length,
        args.do_lower_case, output_prediction_file,
        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
        args.version_2_with_negative, args.null_score_diff_threshold)

    result = _calc_metric_squad(args.predict_file, output_prediction_file)
    os.remove(output_prediction_file)
    os.remove(output_nbest_file)
    os.remove(output_null_log_odds_file)
    return result  # {'exact_match': exact_match, 'f1': f1}
