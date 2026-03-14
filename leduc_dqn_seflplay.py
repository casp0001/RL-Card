import os
import argparse
import copy

import rlcard
from rlcard.agents import DQNAgent, RandomAgent
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
        args.env,
        config={'seed': args.seed}
    )
    eval_env = rlcard.make(
        args.env,
        config={'seed': args.seed}
    )

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Initialise the main DQN agent
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
    )

    # Opponent pool for self-play: starts as a frozen copy of the initial agent
    opponent = copy.deepcopy(agent)

    # For evaluation: agent vs. Random
    eval_env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
    ])

    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):

            # --- Self-play: update opponent snapshot periodically ---
            if episode % args.opponent_update_every == 0 and episode > 0:
                opponent = copy.deepcopy(agent)
                if args.verbose:
                    print(f'\n[Episode {episode}] Opponent snapshot updated.')

            # Set agents for this training episode (agent plays both seats
            # alternately so it sees both positions over time)
            if episode % 2 == 0:
                env.set_agents([agent, opponent])
                learning_player = 0
            else:
                env.set_agents([opponent, agent])
                learning_player = 1

            # Run one episode and feed transitions to the learning agent
            trajectories, payoffs = env.run(is_training=True)

            trajectories = reorganize(trajectories, payoffs)

            # trajectories[i] is the list of (state, action, reward, next_state, done)
            # tuples for seat i.  We only train on the learning agent's transitions.
            for ts in trajectories[learning_player]:
                #print(f'\rEpisode {episode}, training on transition {ts}', end='')
                agent.feed(ts)

            print(f'\rIteration {episode}', end='')

            # Evaluate against Random periodically
            if episode % args.evaluate_every == 0:
                reward = tournament(eval_env, args.num_eval_games)[0]
                logger.log_performance(episode, reward)
                if args.verbose:
                    print(f'\n[Episode {episode}] Avg reward vs Random: {reward:.4f}')

        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Save final model
    save_path = os.path.join(args.log_dir)
    agent.save_checkpoint(save_path)
    print(f'\nModel saved to {save_path}')

    # Plot learning curve
    plot_curve(csv_path, fig_path, 'dqn_selfplay')
    print(f'Learning curve saved to {fig_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN Self-Play example in RLCard")

    # Environment
    parser.add_argument('--env', type=str, default='leduc-holdem',
                        choices=['leduc-holdem', 'limit-holdem', 'no-limit-holdem',
                                 'gin-rummy', 'bridge', 'mahjong', 'uno', 'doudizhu'],
                        help='RLCard environment to train on')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)

    # Training loop
    parser.add_argument('--num_episodes', type=int, default=50_000)
    parser.add_argument('--num_eval_games', type=int, default=2_000)
    parser.add_argument('--evaluate_every', type=int, default=500,
                        help='Evaluate against Random every N episodes')
    parser.add_argument('--opponent_update_every', type=int, default=1_000,
                        help='Freeze a new opponent snapshot every N episodes')

    # DQN hyper-parameters
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replay_memory_size', type=int, default=20_000)
    parser.add_argument('--replay_memory_init_size', type=int, default=100)
    parser.add_argument('--train_every', type=int, default=1,
                        help='Train the network every N steps')
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.1)
    parser.add_argument('--epsilon_decay_steps', type=int, default=20_000)
    parser.add_argument('--mlp_layers', type=int, nargs='+', default=[64, 64],
                        help='Hidden layer sizes, e.g. --mlp_layers 128 128')

    # Misc
    parser.add_argument('--device', type=str, default=None,
                        help='torch device, e.g. "cpu" or "cuda:0" (None = auto)')
    parser.add_argument('--log_dir', type=str,
                        default='experiments/leduc_holdem_dqn_selfplay_result/')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print extra info during training')

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)