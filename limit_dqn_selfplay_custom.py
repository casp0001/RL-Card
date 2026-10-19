import os
import argparse
import copy
import contextlib

import rlcard
from obs_encoding import ENV_IDS
from functions import str2bool, frozen_copy
from eval_panel import Panel, PanelLogger, add_panel_args, parse_members, format_results
from rlcard.agents import RandomAgent
from dqn_agent_relu import DQNAgent 

from rlcard.utils import (
    reorganize,
    set_seed,
    tournament,
    Logger,
    plot_curve,
)
def train(args, on_eval=None):
    ''' Train with the given args. on_eval(episode, panel_results), if given,
    is called after every panel evaluation; training stops early when it
    returns True (used by tune.py). Returns the trained agent. '''
    # Make environments
    env = rlcard.make(
        ENV_IDS[args.obs],
        config={'seed': args.seed}
    )

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Initialise the main DQN agent
    # Limit Hold'em: state_shape is ~72, num_actions is 4 (fold/check/call/raise)
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=args.mlp_layers,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        replay_memory_size=args.replay_memory_size,
        replay_memory_init_size=args.replay_memory_init_size,
        train_every=args.train_every,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        device=args.device,
        activation=args.activation,
        discount_factor=args.discount_factor,
    )

    # Opponent pool for self-play: starts as a frozen copy of the initial agent
    opponent = frozen_copy(agent)

    # Evaluation panel of fixed opponents (see eval_panel.py)
    panel = Panel(parse_members(args.eval_panel), num_deals=args.eval_deals, obs=args.obs)
    panel_logger = PanelLogger(args.log_dir, panel.members)

    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            with open(os.devnull, 'w') as devnull:
                
                with contextlib.redirect_stdout(devnull):

                    # --- Self-play: update opponent snapshot periodically ---
                    if episode % args.opponent_update_every == 0 and episode > 0:
                        opponent = frozen_copy(agent)
                        if args.verbose:
                            print(f'\n[Episode {episode}] Opponent snapshot updated.')

                    # Alternate seats so agent learns both positions
                    if episode % 2 == 0:
                        env.set_agents([agent, opponent])
                        learning_player = 0
                    else:
                        env.set_agents([opponent, agent])
                        learning_player = 1

                    # Run one episode and feed transitions to the learning agent
                    trajectories, payoffs = env.run(is_training=True)
                    trajectories = reorganize(trajectories, payoffs)

                    for ts in trajectories[learning_player]:
                        agent.feed(ts)
                    if args.verbose:
                        print(f'\rIteration {episode}', end='')

            # Evaluate against the panel periodically; performance.csv gets
            # the panel 'min' (robustness score), panel.csv every member
            if episode % args.evaluate_every == 0:
                results = panel.evaluate(agent)
                panel_logger.log(episode, results)
                logger.log_performance(episode, results['min'])
                if on_eval is not None and on_eval(episode, results):
                    break
                print(format_results(results, prefix=f'\n[Episode {episode}] '))

            # Periodic checkpoints (for sweep_exploitability.py)
            if args.save_model and args.checkpoint_every > 0 and episode % args.checkpoint_every == 0 and episode > 0:
                ckpt_dir = os.path.join(args.log_dir, 'checkpoints', f'episode_{episode}')
                os.makedirs(ckpt_dir, exist_ok=True)
                agent.save_checkpoint(ckpt_dir)
                print(f'\nCheckpoint saved at episode {episode} to {ckpt_dir}')

        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Save final model
    if args.save_model:
        agent.save_checkpoint(args.log_dir)
        print(f'\nModel saved to {args.log_dir}')

    # Plot learning curve
    plot_curve(csv_path, fig_path, 'dqn_selfplay_limit_holdem')
    print(f'Learning curve saved to {fig_path}')

    return agent


def build_parser():
    parser = argparse.ArgumentParser("DQN Self-Play — Limit Hold'em (RLCard)")

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)

    # Training loop
    parser.add_argument('--num_episodes', type=int, default=200_000)
    add_panel_args(parser)
    parser.add_argument('--obs', type=str, default='rich', choices=['rich', 'rlcard'],
                        help="Observation encoding: 'rich' (153 values: position, own vs board cards, "
                             "betting sequence; see obs_encoding.py) or RLCard's default 72-value one")
    parser.add_argument('--checkpoint_every', type=int, default=0,
                        help='Save a checkpoint every N episodes (0 = only the final model). '
                             'DQN checkpoints include the replay memory, so they can be large.')
    parser.add_argument('--save_model', type=str2bool, default=True,
                        help='Save checkpoints (tune.py turns this off for trials)')
    parser.add_argument('--evaluate_every', type=int, default=20_000,
                        help='Evaluate against the panel every N episodes')
    parser.add_argument('--opponent_update_every', type=int, default=2_000,
                        help='Freeze a new opponent snapshot every N episodes')

    # DQN hyper-parameters
    parser.add_argument('--mlp_layers', type=int, nargs='+', default=[128, 128],
                        help='Hidden layer sizes, e.g. --mlp_layers 256 256')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replay_memory_size', type=int, default=40_000,
                        help='Larger buffer for longer episodes')
    parser.add_argument('--replay_memory_init_size', type=int, default=100)
    parser.add_argument('--train_every', type=int, default=1)
    parser.add_argument('--discount_factor', type=float, default=0.99,
                        help='Gamma. Hands pay out only at the end, so 1.0 is natural (tuning uses 1.0)')
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.1)
    parser.add_argument('--epsilon_decay_steps', type=int, default=40_000,
                        help='Longer decay for the more complex game')
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['relu', 'tanh'],
                        help='Activation function for the MLP layers')

    # Misc
    parser.add_argument('--device', type=str, default=None,
                        help='torch device, e.g. "cpu" or "cuda:0" (None = auto)')
    parser.add_argument('--log_dir', type=str,
                        default=None)
    parser.add_argument('--verbose', type=str2bool, default=True)

    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    if args.log_dir is None:
        layers_str = '_'.join(str(x) for x in args.mlp_layers)
        args.log_dir = (
            f'experiments/limit_holdem_dqn_selfplay/'
            f'activation{args.activation}_'
            f'lr{args.lr}_bs{args.batch_size}_buf{args.replay_memory_size}_'
            f'layers{layers_str}_eps{args.epsilon_decay_steps}_'
            f'opp{args.opponent_update_every}_seed{args.seed}/'
        )
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)