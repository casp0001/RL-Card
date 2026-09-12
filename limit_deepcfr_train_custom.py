import os
import argparse
import contextlib

import numpy as np
import torch

from honest_agent_limited import HonestAgent

import rlcard
from rlcard.utils.utils import remove_illegal
from deep_cfr_agent_new import DeepCFR

from rlcard.agents import RandomAgent

from functions import warmup, compute_mbb_per_hand, fmt_actions

from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
)


class DeepCFREvalAgent(object):
    ''' Thin wrapper exposing a DQN-style eval_step interface for a DeepCFR agent.

    DeepCFR.eval_step(state) returns only an action id, whereas RLCard's
    tournament/Logger pipeline (and the DQN script) expect a
    (action, info) tuple. This wrapper adapts the signature and also exposes
    action probabilities in `info` so the same metrics can be tracked.
    '''
    use_raw = False

    def __init__(self, agent):
        self._agent = agent
    
    def eval_step(self, state):
        obs = state['obs']
        legal_actions = list(state['legal_actions'].keys())
        action_prob = self._agent.action_probabilities(obs)
        action_prob = remove_illegal(action_prob, legal_actions)
        action_prob /= action_prob.sum()
        action = int(np.random.choice(np.arange(len(action_prob)), p=action_prob))
        info = {'probs': {a: float(action_prob[a]) for a in range(len(action_prob))}}
        return action, info

    def step(self, state):
        action, _ = self.eval_step(state)
        return action


def save_checkpoint(agent, num_players, path):
    ''' Save policy + advantage networks and their optimizers.

    Drop-in replacement for the old `tf.train.Checkpoint(...).write(path)`.
    Stores everything in a single torch file keyed the same way the TF
    checkpoint tracked its objects.
    '''
    torch.save({
        'policy_network': agent._policy_network.state_dict(),
        'optimizer_policy': agent._optimizer_policy.state_dict(),
        **{f'advantage_network_{p}': agent._advantage_networks[p].state_dict()
           for p in range(num_players)},
        **{f'optimizer_advantage_{p}': agent._optimizer_advantages[p].state_dict()
           for p in range(num_players)},
    }, path)


def train(args):
    # Make environments
    env = rlcard.make(
        'limit-holdem',
        config={'seed': args.seed, 
                'allow_step_back': True}
    )
    eval_env = rlcard.make(
        'limit-holdem',
        config={'seed': args.seed, 
                'allow_step_back': True}
    )

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Deep CFR (PyTorch) — traverses `env` directly; no session/graph needed.
    agent = DeepCFR(
        env=env,
        policy_network_layers=tuple(args.policy_layers),
        advantage_network_layers=tuple(args.advantage_layers),
        num_traversals=args.num_traversals,
        num_step=args.num_step,
        learning_rate=args.lr,
        batch_size_advantage=args.batch_size_advantage,
        batch_size_strategy=args.batch_size_strategy,
        memory_capacity=args.memory_capacity,
    )

    # Number of players, used for checkpoint keys below.
    num_players = env.num_players

    # Eval wrapper so the DeepCFR agent plugs into the RLCard pipeline.
    eval_agent = DeepCFREvalAgent(agent)

    # For evaluation: agent vs. a fixed honest opponent (mirrors DQN script)
    eval_env.set_agents([
        eval_agent,
        RandomAgent(num_actions=env.num_actions),
    ])

    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    # One Deep CFR iteration: traverse the game tree for
                    # each player and update advantage + strategy nets.
                    _, avg_adv_loss, policy_loss = agent.train()
                    if args.verbose:
                        print(f'\rIteration {episode}', end='')

            # Evaluate against the honest opponent periodically
            if episode % args.evaluate_every == 0:
                action_counts = {0: 0, 1: 0, 2: 0}
                total_steps = 0
                reward = tournament(eval_env, args.num_eval_games)[0]
                original_eval_step = eval_agent.eval_step

                def counting_eval_step_avg(state):
                    action, info = original_eval_step(state)
                    action_counts[action] = action_counts.get(action, 0) + 1
                    nonlocal total_steps
                    total_steps += 1
                    return action, info
                eval_agent.eval_step = counting_eval_step_avg
                tournament(eval_env, 500)  # small sample just for action counts
                eval_agent.eval_step = original_eval_step
                mbb = compute_mbb_per_hand(reward)
                logger.log_performance(episode, mbb)
                print(f'mbb/h: {mbb:.2f}')
                print(f'[{episode}] avg_policy: {reward:.4f}')
                print(f'  best_response actions: {fmt_actions(action_counts, total_steps)}')
                print(f'  adv_loss: {avg_adv_loss:.6f}  policy_loss: {policy_loss:.6f}')
                if episode % 10 == 0 and episode > 0:
                    ckpt_dir = f'{args.log_dir}checkpoints/episode_{episode}'
                    os.makedirs(ckpt_dir, exist_ok=True)
                    save_checkpoint(agent, num_players, f'{ckpt_dir}/model.ckpt')
                    logger.csv_file.flush()
                    plot_curve(logger.csv_path, logger.fig_path, 'deepcfr_limit_holdem')
                    print(f'\nCheckpoint saved at episode {episode} to {ckpt_dir}')
            print(f'{episode}/{args.num_episodes} complete.', end='\r')

        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Save final model
    save_checkpoint(agent, num_players, f'{args.log_dir}/model.ckpt')
    print(f'\nModel saved to {args.log_dir}')

    # Plot learning curve
    plot_curve(csv_path, fig_path, 'deepcfr_limit_holdem')
    print(f'Learning curve saved to {fig_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Deep CFR — Limit Hold'em (RLCard)")

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)

    # Training loop
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Number of Deep CFR iterations')
    parser.add_argument('--num_eval_games', type=int, default=4_000)
    parser.add_argument('--evaluate_every', type=int, default=2,
                        help='Evaluate against the honest opponent every N iterations')

    # Deep CFR hyper-parameters
    parser.add_argument('--policy_layers', type=int, nargs='+', default=[128, 128, 128],
                        help='Hidden layer sizes for the strategy network')
    parser.add_argument('--advantage_layers', type=int, nargs='+', default=[128, 128, 128],
                        help='Hidden layer sizes for the advantage networks')
    parser.add_argument('--num_traversals', type=int, default=500,
                        help='Number of game-tree traversals per iteration')
    parser.add_argument('--num_step', type=int, default=400,
                        help='SGD steps per network update')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size_advantage', type=int, default=2048)
    parser.add_argument('--batch_size_strategy', type=int, default=2048)
    parser.add_argument('--memory_capacity', type=int, default=10_000_000)

    # Misc
    parser.add_argument('--device', type=str, default=None,
                        help='Unused for the torch Deep CFR agent; kept for parity')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--verbose', type=bool, default=True)

    args = parser.parse_args()
    if args.log_dir is None:
        pol_str = '_'.join(str(x) for x in args.policy_layers)
        adv_str = '_'.join(str(x) for x in args.advantage_layers)
        args.log_dir = (
            f'experiments/deepcfr/'
            f'lr{args.lr}_trav{args.num_traversals}_step{args.num_step}_'
            f'pol{pol_str}_adv{adv_str}_'
            f'layers{args.policy_layers}_{args.advantage_layers}_'
            f'mem{args.memory_capacity}_'
            f'basadv{args.batch_size_advantage}_bsstr{args.batch_size_strategy}_'
            f'seed{args.seed}/'
        )
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)
