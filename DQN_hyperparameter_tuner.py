import os 
import argparse
import itertools
import subprocess
import sys

import csv

GRID = {
    'lr':                   [1e-3, 1e-4],
    'batch_size':           [32, 64],
    'mlp_layers':           [[128, 128], [256, 256], [128, 128, 128]],
    'epsilon_decay_steps':  [20_000, 40_000],
    'activation':         ['relu', 'tanh'],
    'opponent_update_every': [1_000],
}

FIXED = {
    'num_episodes':         50_000,
    'num_eval_games':       1_000,
    'evaluate_every':       1_000,
    'seed':                 42,
    'verbose':              False,
}

def grid_configs(grid):
    keys, values = zip(*grid.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def build_cmd(cfg):
    cmd = ['python', 'limit_dqn_selfplay_custom.py']
    for k, v in {**cfg, **FIXED}.items():
        if isinstance(v, list):
            cmd += [f'--{k}'] + [str(x) for x in v]
        else:
            cmd += [f'--{k}', str(v)]
    return cmd


def infer_log_dir(cfg):
    layers_str = '_'.join(str(x) for x in cfg['mlp_layers'])
    return (
        f"experiments/limit_holdem_dqn_selfplay/"
        f'activation{cfg["activation"]}_'
        f"lr{cfg['lr']}_bs{cfg['batch_size']}_buf40000_"
        f"layers{layers_str}_eps{cfg['epsilon_decay_steps']}_"
        f"opp{cfg['opponent_update_every']}_seed{FIXED['seed']}/"
    )


def read_last_reward(log_dir):
    csv_path = os.path.join(log_dir, 'performance.csv')
    try:
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            last = None
            for row in reader:
                if len(row) >= 2:  # skip empty or malformed lines
                    last = float(row[1])
        return last
    except FileNotFoundError:
        return None


configs = list(grid_configs(GRID))
print(f'Total runs: {len(configs)}\n')

results = []
for i, cfg in enumerate(configs, 1):
    cmd = build_cmd(cfg)
    log_dir = infer_log_dir(cfg)
    print(f'[{i}/{len(configs)}] Running: {" ".join(cmd)}')

    output_lines = []
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)
    process.wait()

    last = read_last_reward(log_dir)
    results.append((last, cfg))
    print(f'  Done.\n')

results.sort(reverse=True, key=lambda x: (x[0] is not None, x[0]))
print('\n=== Results (best to worst) ===')
for rank, (reward, cfg) in enumerate(results, 1):
    cfg_str = ', '.join(f'{k}={v}' for k, v in cfg.items())
    reward_str = f'{reward:.4f}' if reward is not None else 'N/A'
    print(f'{rank}. reward={reward_str}  |  {cfg_str}')