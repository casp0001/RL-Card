import os
import argparse
import copy
import contextlib

from honest_agent_limited import HonestAgent

import rlcard
from rlcard.agents import RandomAgent
from dqn_agent_relu import DQNAgent 

from rlcard.utils import (
    reorganize,
    set_seed,
    tournament,
    Logger,
    plot_curve,
)


def train(args):
    # make environments
    env = rlcard.make(
        'limit-holdem',
        config={'seed': args.seed}
    )
    eval_env = rlcard.make(
        'limit-holdem',
        config={'seed': args.seed}
    )

    set_seed(args.seed)

    # initialise the main DQN agent
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

    # create agent copy for opponent
    opponent = copy.deepcopy(agent)

    # evaluation environemtn, agent vs honest "rational" opponent
    eval_env.set_agents([
        agent,
        HonestAgent(num_actions=env.num_actions),
    ])

    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):

                    # periodic opponent snapshot update
                    if episode % args.opponent_update_every == 0 and episode > 0:
                        opponent = copy.deepcopy(agent)
                        if args.verbose:
                            print(f'\n[Episode {episode}] Opponent snapshot updated.')

                    # alternate seats so agent learns both positions
                    if episode % 2 == 0:
                        env.set_agents([agent, opponent])
                        learning_player = 0
                    else:
                        env.set_agents([opponent, agent])
                        learning_player = 1

                    # feed transitions to the learning agent
                    trajectories, payoffs = env.run(is_training=True)
                    trajectories = reorganize(trajectories, payoffs)

                    for ts in trajectories[learning_player]:
                        agent.feed(ts)
                    if args.verbose:
                        print(f'\rIteration {episode}', end='')

            # periodic evalutaion against honest opponent
            if episode % args.evaluate_every == 0:
                reward = tournament(eval_env, args.num_eval_games)[0]
                logger.log_performance(episode, reward)
                if args.verbose == True:
                    print(f'\n[Episode {episode}] Avg reward vs Opponent: {reward:.4f}')

        csv_path, fig_path = logger.csv_path, logger.fig_path


    agent.save_checkpoint(args.log_dir)
    print(f'\nModel saved to {args.log_dir}')


    plot_curve(csv_path, fig_path, 'dqn_selfplay_limit_holdem')
    print(f'Learning curve saved to {fig_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN Self-Play — Limit Hold'em (RLCard)")

    # seed for reproducibility
    parser.add_argument('--seed', type=int, default=42)

    # training loop hyper-parameters
    parser.add_argument('--num_episodes', type=int, default=200_000)
    parser.add_argument('--num_eval_games', type=int, default=2_000)
    parser.add_argument('--evaluate_every', type=int, default=1_000)
    parser.add_argument('--opponent_update_every', type=int, default=2_000)

    # DQN hyper-parameters
    parser.add_argument('--mlp_layers', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replay_memory_size', type=int, default=40_000)
    parser.add_argument('--replay_memory_init_size', type=int, default=100)
    parser.add_argument('--train_every', type=int, default=1)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.1)
    parser.add_argument('--epsilon_decay_steps', type=int, default=40_000)
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'tanh'])

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