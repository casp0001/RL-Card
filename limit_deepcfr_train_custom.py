import os
import argparse
import contextlib

import numpy as np
import torch


import rlcard
from rlcard.utils.utils import remove_illegal
from deep_cfr_agent_new import DeepCFR


from functions import str2bool
from eval_panel import Panel, PanelLogger, add_panel_args, parse_members, format_results

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


def train(args, on_eval=None):
    ''' Train with the given args. on_eval(episode, panel_results), if given,
    is called after every panel evaluation; training stops early when it
    returns True (used by tune.py). Returns the trained agent. '''
    # Make environments
    env = rlcard.make(
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

    # Evaluation panel of fixed opponents (see eval_panel.py)
    panel = Panel(parse_members(args.eval_panel), num_deals=args.eval_deals)
    panel_logger = PanelLogger(args.log_dir, panel.members)

    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    # One Deep CFR iteration: traverse the game tree for
                    # each player and update advantage + strategy nets.
                    _, avg_adv_loss, policy_loss = agent.train()
                    if args.verbose:
                        print(f'\rIteration {episode}', end='')

            # Evaluate the average policy against the panel periodically;
            # performance.csv gets the panel 'min', panel.csv every member
            if episode % args.evaluate_every == 0:
                results = panel.evaluate(eval_agent)
                panel_logger.log(episode, results)
                logger.log_performance(episode, results['min'])
                if on_eval is not None and on_eval(episode, results):
                    break
                print(format_results(results, prefix=f'[Iteration {episode}] '))
                print(f'  last-step weighted losses  adv: {avg_adv_loss:.6f}  policy: {policy_loss:.6f}')

            if args.save_model and episode % args.checkpoint_every == 0 and episode > 0:
                ckpt_dir = f'{args.log_dir}checkpoints/episode_{episode}'
                os.makedirs(ckpt_dir, exist_ok=True)
                save_checkpoint(agent, num_players, f'{ckpt_dir}/model.ckpt')
                logger.csv_file.flush()
                plot_curve(logger.csv_path, logger.fig_path, 'deepcfr_limit_holdem')
                print(f'\nCheckpoint saved at episode {episode} to {ckpt_dir}')
            print(f'{episode}/{args.num_episodes} complete.', end='\r')

        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Save final model
    if args.save_model:
        save_checkpoint(agent, num_players, f'{args.log_dir}/model.ckpt')
        print(f'\nModel saved to {args.log_dir}')

    # Plot learning curve
    plot_curve(csv_path, fig_path, 'deepcfr_limit_holdem')
    print(f'Learning curve saved to {fig_path}')

    return eval_agent


def build_parser():
    parser = argparse.ArgumentParser("Deep CFR — Limit Hold'em (RLCard)")

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)

    # Training loop
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Number of Deep CFR iterations')
    add_panel_args(parser)
    parser.add_argument('--save_model', type=str2bool, default=True,
                        help='Save checkpoints (tune.py turns this off for trials)')
    parser.add_argument('--evaluate_every', type=int, default=2,
                        help='Evaluate against the panel every N iterations')
    parser.add_argument('--checkpoint_every', type=int, default=10,
                        help='Save a checkpoint every N iterations')

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
    parser.add_argument('--verbose', type=str2bool, default=True)

    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
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
