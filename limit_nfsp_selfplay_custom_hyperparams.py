import os
import argparse
import copy
import contextlib

import rlcard
from functions import fmt_actions, str2bool
from eval_panel import Panel, PanelLogger, add_panel_args, parse_members, format_results
from nfsp_agent_XPU import NFSPAgent
from rlcard.utils import (
    reorganize,
    set_seed,
    tournament,
    Logger,
    plot_curve,
)
def warmup(agent, args):
    from honest_agent_limited import HonestAgent
    print("Warming up agent against HonestAgent...")
    warmup_env = rlcard.make('limit-holdem', config={'seed': args.seed})
    warmup_env.set_agents([agent, HonestAgent(num_actions=warmup_env.num_actions)])
    for ep in range(50_000):
        agent.sample_episode_policy()
        trajectories, payoffs = warmup_env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)
        for ts in trajectories[0]:
            agent.feed(ts)
    print("Warmup complete.")


def train(args, on_eval=None):
    ''' Train with the given args. on_eval(episode, panel_results), if given,
    is called after every panel evaluation; training stops early when it
    returns True (used by tune.py). Returns the trained agent. '''
    # make environments
    env = rlcard.make(
        'limit-holdem',
        config={'seed': args.seed}
    )
    set_seed(args.seed)
    # initialise the main NFSP agent
    #print(env.num_actions, env.state_shape)
    agent = NFSPAgent(
        num_actions=env.num_actions,   
        state_shape=[72],    
        hidden_layers_sizes = args.hidden_layers_sizes,
        reservoir_buffer_capacity=args.reservoir_buffer_capacity,
        anticipatory_param=args.anticipatory_param,
        batch_size=args.batch_size,
        train_every=args.train_every,
        rl_learning_rate=args.rl_learning_rate,
        sl_learning_rate=args.sl_learning_rate,
        min_buffer_size_to_learn=args.min_buffer_size_to_learn,
        q_replay_memory_size=args.q_replay_memory_size,
        q_replay_memory_init_size=args.q_replay_memory_init_size,
        q_update_target_estimator_every=args.q_update_target_estimator_every,
        q_discount_factor=args.q_discount_factor,
        q_epsilon_start=args.q_epsilon_start,
        q_epsilon_end=args.q_epsilon_end,
        q_epsilon_decay_steps=args.q_epsilon_decay_steps,
        q_batch_size=args.q_batch_size,
        q_train_every=args.q_train_every,
        q_mlp_layers=args.q_mlp_layers,
        q_activation=args.q_activation,
        optimizer=args.optimizer,
        device=args.device,
    )
    #warmup(agent, args)  # optional warmup against honest opponent before self-play
    # create agent copy for opponent
    opponent = copy.deepcopy(agent)
    # Evaluation panel of fixed opponents (see eval_panel.py)
    panel = Panel(parse_members(args.eval_panel), num_deals=args.eval_deals)
    panel_logger = PanelLogger(args.log_dir, panel.members)
    panel_logger_br = (PanelLogger(args.log_dir, panel.members, filename='panel_br')
                       if args.eval_best_response else None)

    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            agent.sample_episode_policy()
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
                        seat = 0
                    else:
                        env.set_agents([opponent, agent])
                        seat = 1

                    # resample the snapshot's mode too, otherwise it keeps the mode it
                    # was copied in (pure best response w.p. eta) for the whole window
                    opponent.sample_episode_policy()
                    trajectories, payoffs = env.run(is_training=True)
                    trajectories = reorganize(trajectories, payoffs)

                    for ts in trajectories[seat]:  
                        agent.feed(ts)
                if args.verbose:
                    print(f'\rIteration {episode}', end='')

            # Evaluate the average policy (what NFSP plays) against the panel
            # periodically; performance.csv gets the panel 'min', panel.csv
            # every member. Optionally also the best-response network.
            if episode % args.evaluate_every == 0:
                results = panel.evaluate(agent, nfsp_policy='average_policy')
                panel_logger.log(episode, results)
                logger.log_performance(episode, results['min'])
                if on_eval is not None and on_eval(episode, results):
                    break
                print(format_results(results, prefix=f'[Episode {episode}] avg policy: '))
                if args.eval_best_response:
                    results_br = panel.evaluate(agent, nfsp_policy='best_response')
                    panel_logger_br.log(episode, results_br)
                    print(format_results(results_br, prefix=f'[Episode {episode}] best response: '))
            if args.save_model and episode % 1_000_000 == 0 and episode > 0:
                    os.makedirs(f'{args.log_dir}checkpoints/episode_{episode}', exist_ok=True)
                    agent.save_checkpoint(f'{args.log_dir}/checkpoints/episode_{episode}')
                    print(f'\nCheckpoint saved at episode {episode} to {args.log_dir}/checkpoints/episode_{episode}')
        csv_path, fig_path = logger.csv_path, logger.fig_path


    if args.save_model:
        agent.save_checkpoint(args.log_dir)
        print(f'\nModel saved to {args.log_dir}')


    plot_curve(csv_path, fig_path, 'nfsp_selfplay_limit_holdem')
    print(f'Learning curve saved to {fig_path}')

    return agent


def build_parser():
    parser = argparse.ArgumentParser("NFSP Self-Play — Limit Hold'em (RLCard)")

    # seed for reproducibility
    parser.add_argument('--seed', type=int, default=42)

    # training loop hyper-parameters
    parser.add_argument('--num_episodes', type=int, default=2_000_000)
    add_panel_args(parser)
    parser.add_argument('--save_model', type=str2bool, default=True,
                        help='Save checkpoints (tune.py turns this off for trials)')
    parser.add_argument('--eval_best_response', type=str2bool, default=False,
                        help='Also evaluate the best-response (DQN) network (panel_br.csv)')
    parser.add_argument('--evaluate_every', type=int, default=10_000)
    parser.add_argument('--opponent_update_every', type=int, default=50_000)

    # NFSP hyper-parameters
    parser.add_argument('--hidden_layers_sizes', type=int, nargs='+', default=[256, 256, 256])
    parser.add_argument('--reservoir_buffer_capacity', type=int, default = 1_500_000)
    parser.add_argument('--anticipatory_param', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_every', type=int, default=256)
    parser.add_argument('--rl_learning_rate', type=float, default=0.02)
    parser.add_argument('--sl_learning_rate', type=float, default=0.01)
    parser.add_argument('--min_buffer_size_to_learn', type=int, default=10000)
    parser.add_argument('--q_replay_memory_size', type=int, default=200_000)
    parser.add_argument('--q_replay_memory_init_size', type=int, default=10_000)
    parser.add_argument('--q_update_target_estimator_every', type=int, default=1000)
    parser.add_argument('--q_discount_factor', type=float, default=0.99)
    parser.add_argument('--q_epsilon_start', type=float, default=0.08)
    parser.add_argument('--q_epsilon_end', type=float, default=0.01)
    parser.add_argument('--q_epsilon_decay_steps', type=int, default=int(3e6))
    parser.add_argument('--q_batch_size', type=int, default=256)
    parser.add_argument('--q_train_every', type=int, default=128)
    parser.add_argument('--q_mlp_layers', type=int, nargs='+', default=[256, 256, 256])
    parser.add_argument('--q_activation', type=str, default='tanh', choices=['tanh', 'relu'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='optimizer for both the RL (DQN) and SL networks')
    

    # misc
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--log_dir', type=str,
                        default=None)
    parser.add_argument('--verbose', type=str2bool, default=True)

    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    # log directory dymanic creation 
    if args.log_dir is None:
        layers_str = '_'.join(str(x) for x in args.q_mlp_layers)
        args.log_dir = (
            f'experiments/nfsp/'
            f'res{args.reservoir_buffer_capacity}_anticip{args.anticipatory_param}_batch{args.batch_size}_t_e{args.train_every}_'
            f'lr{args.rl_learning_rate}_sl{args.sl_learning_rate}_bs{args.batch_size}_buf{args.q_replay_memory_size}_'
            f'lrs_{layers_str}'
            f'eps_s{args.q_epsilon_start}_eps_e{args.q_epsilon_end}_'
            f'eps_d{args.q_epsilon_decay_steps}_'
            f'opp{args.opponent_update_every}_seed{args.seed}/'
        )
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)
