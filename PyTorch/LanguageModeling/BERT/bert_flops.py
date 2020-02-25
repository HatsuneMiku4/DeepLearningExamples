import argparse
import os

from apex.normalization import FusedLayerNorm
from thop import profile, clever_format
from torch.nn import Embedding, Softmax
from torch.utils.data import RandomSampler, DataLoader

from modeling import BertConfig, BertForPreTraining, LinearActivation
from run_pretraining import pretraining_dataset

args = argparse.Namespace(
    input_dir='data/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus/',
    max_predictions_per_seq=80,
    train_batch_size=1,
    n_gpu=1,
    config_file='data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_config.json',
    checkpoint_activations=False,
)

assert args.train_batch_size == 1
assert args.n_gpu == 1

# Load dataset
data_file = None
for f in os.listdir(args.input_dir):
    full_path = os.path.join(args.input_dir, f)
    if not os.path.isfile(full_path): continue
    if 'training' not in f: continue
    data_file = full_path

train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(
    train_data, sampler=train_sampler,
    batch_size=args.train_batch_size * args.n_gpu,  # batch_size set to 1
    num_workers=4, pin_memory=True
)

# Sample one batch from the training set
batch = [t.cuda() for t in next(iter(train_dataloader))]
input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch

print(f'input_ids: {input_ids.shape}')
print(f'segment_ids: {segment_ids.shape}')
print(f'input_mask: {input_mask.shape}')
print(f'masked_lm_labels: {masked_lm_labels.shape}')
print(f'next_sentence_labels: {next_sentence_labels.shape}')

# Load model
config = BertConfig.from_json_file(args.config_file)

# We skip padding for consistency with the HuggingFace repository
# if config.vocab_size % 8 != 0:
#     config.vocab_size += 8 - (config.vocab_size % 8)

# noinspection PyUnresolvedReferences
model = BertForPreTraining(config).cuda()

loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
             masked_lm_labels=masked_lm_labels, next_sentence_label=next_sentence_labels,
             checkpoint_activations=args.checkpoint_activations)

flops, params = clever_format(profile(
    model, inputs=batch, custom_ops={
        Embedding: None,  # TODO: custom operator: Embedding
        FusedLayerNorm: None,  # TODO: custom operator: FusedLayerNorm
        LinearActivation: None,  # TODO: custom operator: LinearActivation
        Softmax: None,  # TODO: custom operator: Softmax
    }
), '%.3f')
print(f'FLOPs: {flops}')  # FLOPs: 81.970G
print(f'Params: {params}')  # Params: 133.517M
