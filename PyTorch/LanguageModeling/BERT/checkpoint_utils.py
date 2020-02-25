import argparse

import torch

from admm_manager_v2 import PruningPhase

parser = argparse.ArgumentParser(
    description='Simulate a checkpoint saved at the last step of ADMM pruning')
parser.add_argument("--input_checkpoint", default=None, type=str, required=True,
                    help="Path to the input checkpoint.")
parser.add_argument("--output_checkpoint", default=None, type=str, required=True,
                    help="Path to the output checkpoint.")
args = parser.parse_args()

print(f'loading checkpoint at {args.input_checkpoint}...')
checkpoint = torch.load(args.input_checkpoint, map_location='cpu')
# e.g. 'model', 'admm', 'config', 'optimizer', 'amp', 'timer'
print(f'checkpoint keys: {list(checkpoint.keys())}')
# e.g. {'cur_phase': 'masked_retrain', 'cur_rho': None, 'cur_step': 191999}
print(f'old timer status: {checkpoint["timer"]}')

assert checkpoint['timer']['cur_phase'] == PruningPhase.admm.name

config = checkpoint['config']
print(f'training configurations: \n{config}')

# see: ProximalADMMPruningManager._calc_current_rho
final_rho = config['initial_rho'] * 10 ** (config['rho_num'] - 1)
total_admm_steps = config['admm_steps'] - 1
checkpoint['timer']['cur_rho'] = final_rho
checkpoint['timer']['cur_step'] = total_admm_steps

print(f'new timer status: {checkpoint["timer"]}')
print(f'saving checkpoint to {args.output_checkpoint}...')

torch.save(checkpoint, args.output_checkpoint)
