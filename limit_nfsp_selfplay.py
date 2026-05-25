import os
import argparse
import copy
import contextlib

from honest_agent_limited import HonestAgent
import rlcard
from rlcard.agents import NFSPAgent
from rlcard.utils import (
    reorganize,
    set_seed,
    tournament,
    Logger,
    plot_curve,
)

def train(args):
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser("NFSP Self-Play — Limit Hold'em (RLCard)")

    # seed for reproducibility
    parser.add_argument('--seed', type=int, default=42)

    # training loop hyper-parameters
    parser.add_argument('--num_episodes', type=int, default=200_000)
    parser.add_argument('--num_eval_games', type=int, default=2_000)
    parser.add_argument('--evaluate_every', type=int, default=1_000)
    parser.add_argument('--opponent_update_every', type=int, default=2_000)

    # NFSP hyper-parameters
    parser.add_argument('--reservoir_buffer_capacity', type=int, default = 20000)
    parser.add_argument('--anticipatory_param', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_every', type=int, default=1)
    parser.add_argument('--rl_learning_rate', type=float, default=0.1)
    parser.add_argument('--sl_learning_rate', type=float, default=0.005)
    parser.add_argument('--min_buffer_size_to_learn', type=int, default=100)
    parser.add_argument('--q_replay_memory_size', type=int, default=20000)
    parser.add_argument('--q_replay_memory_init_size', type=int, default=100)
    parser.add_argument('--q_update_target_estimator_every', type=int, default=1000)
    parser.add_argument('--q_discount_factor', type=float, default=0.99)
    parser.add_argument('--q_epsilon_start', type=float, default=0.06)
    parser.add_argument('--q_epsilon_end', type=float, default=0)
    parser.add_argument('--q_epsilon_decay_steps', type=int, default=int(1e6))
    parser.add_argument('--q_batch_size', type=int, default=32)
    parser.add_argument('--q_train_every', type=int, default=1)
    parser.add_argument('--q_mlp_layers', type=int, nargs='+', default=[128, 128])
    

    # misc
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--log_dir', type=str,
                        default=None)
    parser.add_argument('--verbose', type=bool, default=True)

    args = parser.parse_args()
    # log directory dymanic creation 
    if args.log_dir is None:
        layers_str = '_'.join(str(x) for x in args.mlp_layers)
        args.log_dir = (
            f'experiments/limit_holdem_dqn_selfplay/'
            f'activation{args.activation}_'
            f'lr{args.lr}_bs{args.batch_size}_buf{args.replay_memory_size}_'
            f'layers_{layers_str}'
            f'eps_start{args.epsilon_start}_eps_end{args.epsilon_end}_'
            f'eps_decay{args.epsilon_decay_steps}_'
            f'opp{args.opponent_update_every}_seed{args.seed}/'
        )
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)
