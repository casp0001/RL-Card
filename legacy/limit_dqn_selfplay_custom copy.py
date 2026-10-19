import os
import argparse
import copy
import contextlib

from honest_agent_limited import HonestAgent

import rlcard
from rlcard.agents import RandomAgent
from dqn_agent_relu import DQNAgent 

from functions import warmup, compute_mbb_per_hand, fmt_actions, str2bool

from rlcard.utils import (
    reorganize,
    set_seed,
    tournament,
    Logger,
    plot_curve,
)

def train(args):
    # Make environments
    env = rlcard.make(
        'limit-holdem',
        config={'seed': args.seed}
    )
    eval_env = rlcard.make(
        'limit-holdem',
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
    )

    # Opponent pool for self-play: starts as a frozen copy of the initial agent
    opponent = copy.deepcopy(agent)

    # For evaluation: agent vs. Random
    eval_env.set_agents([
        agent,
        HonestAgent(num_actions=env.num_actions),
    ])

    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            with open(os.devnull, 'w') as devnull:
                
                with contextlib.redirect_stdout(devnull):

                    # --- Self-play: update opponent snapshot periodically ---
                    if episode % args.opponent_update_every == 0 and episode > 0:
                        opponent = copy.deepcopy(agent)
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

            # Evaluate against Random periodically
            if episode % args.evaluate_every == 0:
                action_counts = {0: 0, 1: 0, 2: 0}
                total_steps = 0
                reward = tournament(eval_env, args.num_eval_games)[0]
                original_eval_step = agent.eval_step
                def counting_eval_step_avg(state):
                    action, info = original_eval_step(state)
                    action_counts[action] = action_counts.get(action, 0) + 1
                    nonlocal total_steps
                    total_steps += 1
                    return action, info                
                agent.eval_step = counting_eval_step_avg
                tournament(eval_env, 500)  # small sample just for action counts
                agent.eval_step = original_eval_step
                mbb = compute_mbb_per_hand(reward)
                logger.log_performance(episode, mbb)
                print(f'mbb/h: {mbb:.2f}')
                print(f'[{episode}] avg_policy: {reward:.4f}')
                print(f'  best_response actions: {fmt_actions(action_counts, total_steps)}')
                if episode % 100_000 == 0 and episode > 0:
                    os.makedirs(f'{args.log_dir}checkpoints/episode_{episode}', exist_ok=True)
                    agent.save_checkpoint(f'{args.log_dir}/checkpoints/episode_{episode}')
                    logger.csv_file.flush()
                    plot_curve(logger.csv_path, logger.fig_path, 'dqn_selfplay_limit_holdem')
                    print(f'\nCheckpoint saved at episode {episode} to {args.log_dir}/checkpoints/episode_{episode}')

        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Save final model
    agent.save_checkpoint(args.log_dir)
    print(f'\nModel saved to {args.log_dir}')

    # Plot learning curve
    plot_curve(csv_path, fig_path, 'dqn_selfplay_limit_holdem')
    print(f'Learning curve saved to {fig_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN Self-Play — Limit Hold'em (RLCard)")

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)

    # Training loop
    parser.add_argument('--num_episodes', type=int, default=1_000_000)
    parser.add_argument('--num_eval_games', type=int, default=4_000)
    parser.add_argument('--evaluate_every', type=int, default=50_000,
                        help='Evaluate against Random every N episodes')
    parser.add_argument('--opponent_update_every', type=int, default=4_000,
                        help='Freeze a new opponent snapshot every N episodes')

    # DQN hyper-parameters
    parser.add_argument('--mlp_layers', type=int, nargs='+', default=[128, 128, 128],
                        help='Hidden layer sizes, e.g. --mlp_layers 256 256')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replay_memory_size', type=int, default=100_000,
                        help='Larger buffer for longer episodes')
    parser.add_argument('--replay_memory_init_size', type=int, default=100)
    parser.add_argument('--train_every', type=int, default=1)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.05)
    parser.add_argument('--epsilon_decay_steps', type=int, default=700_000,
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

    args = parser.parse_args()
    if args.log_dir is None:
        layers_str = '_'.join(str(x) for x in args.mlp_layers)
        args.log_dir = (
            f'experiments/dqn/'
            f'activation{args.activation}_'
            f'lr{args.lr}_bs{args.batch_size}_buf{args.replay_memory_size}_'
            f'layers{layers_str}_eps{args.epsilon_decay_steps}_'
            f'opp{args.opponent_update_every}_seed{args.seed}/'
        )
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)