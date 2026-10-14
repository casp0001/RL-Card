import os
import argparse

import torch
import matplotlib.pyplot as plt

import rlcard
from rlcard.utils import tournament

# The local DQN restores the activation stored in the checkpoint (older
# checkpoints from limit_dqn_selfplay_custom.py were all ReLU). rlcard's own
# DQNAgent always builds a Tanh network, so loading these checkpoints into it
# silently computes the wrong function.
from dqn_agent_XPU import DQNAgent
from functions import compute_mbb_per_hand


def load_dqn(path):
    return DQNAgent.from_checkpoint(torch.load(path, map_location='cpu', weights_only=False))


parser = argparse.ArgumentParser('Head-to-head between two DQN checkpoints')
# Both defaults were trained by limit_dqn_selfplay_custom.py. Before the
# activation fix that script always used ReLU, so the "activationtanh" run is
# a ReLU network too; the two differ only in the run, not the activation.
parser.add_argument('--ckpt_a', default='experiments/limit_holdem_dqn_selfplay/activationrelu_lr0.001_bs32_buf40000_layers128_128_128_eps40000_opp1000_seed42/checkpoint_dqn.pt')
parser.add_argument('--ckpt_b', default='experiments/limit_holdem_dqn_selfplay/activationtanh_lr0.001_bs32_buf40000_layers128_128_128_eps40000_opp1000_seed42/checkpoint_dqn.pt')
parser.add_argument('--label_a', default='DQN run A ("relu" dir)')
parser.add_argument('--label_b', default='DQN run B ("tanh" dir, actually ReLU)')
parser.add_argument('--num_games', type=int, default=10_000)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

env = rlcard.make('limit-holdem', config={'seed': args.seed})
env.set_agents([load_dqn(args.ckpt_a), load_dqn(args.ckpt_b)])

payoffs = tournament(env, num=args.num_games)
mbb = [compute_mbb_per_hand(p) for p in payoffs]

print(f'{args.label_a}: {mbb[0]:+.1f} mbb/h')
print(f'{args.label_b}: {mbb[1]:+.1f} mbb/h')

agents = [args.label_a, args.label_b]
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(agents, mbb, color=['steelblue', 'tomato'])
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.bar_label(bars, fmt='{:+.1f}', padding=4)
ax.set_ylabel('Win rate (mbb/h)')
ax.set_title(f'Tournament Results ({args.num_games} games)')
plt.tight_layout()
os.makedirs('head_to_head_template', exist_ok=True)
plt.savefig('head_to_head_template/tournament.png', dpi=150)
plt.show()
