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
def warmup(agent, args):
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


def compute_mbb_per_hand(avg_payoff, big_blind=2):
    return (avg_payoff / big_blind) * 1000

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
    )
    #warmup(agent, args)  # optional warmup against honest opponent before self-play
    # create agent copy for opponent
    opponent = copy.deepcopy(agent)
     # evaluation environemtn, agent vs honest "rational" opponent
    eval_env.set_agents([
        agent,
        HonestAgent(num_actions=env.num_actions),
    ])

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

                    trajectories, payoffs = env.run(is_training=True)
                    trajectories = reorganize(trajectories, payoffs)

                    for ts in trajectories[seat]:  
                        agent.feed(ts)
                if args.verbose:
                    print(f'\rIteration {episode}', end='')

            # periodic evalutaion against honest opponent
            if episode % args.evaluate_every == 0:
                # eval average policy (default)
                agent.evaluate_with = 'average_policy'
                reward_avg = tournament(eval_env, args.num_eval_games)[0]
                
                # eval best response policy
                agent.evaluate_with = 'best_response'
                reward_br = tournament(eval_env, args.num_eval_games)[0]
                
                mbb = compute_mbb_per_hand(reward_avg)
                logger.log_performance(episode, mbb)
                print(f'mbb/h: {mbb:.2f}')
                print(f'[{episode}] avg_policy: {reward_avg:.4f} | best_response: {reward_br:.4f}')
            if episode % 1_000_000 == 0 and episode > 0:
                    os.makedirs(f'{args.log_dir}checkpoints/episode_{episode}', exist_ok=True)
                    agent.save_checkpoint(f'{args.log_dir}/checkpoints/episode_{episode}')
                    print(f'\nCheckpoint saved at episode {episode} to {args.log_dir}/checkpoints/episode_{episode}')
        csv_path, fig_path = logger.csv_path, logger.fig_path


    agent.save_checkpoint(args.log_dir)
    print(f'\nModel saved to {args.log_dir}')


    plot_curve(csv_path, fig_path, 'nfsp_selfplay_limit_holdem')
    print(f'Learning curve saved to {fig_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("NFSP Self-Play — Limit Hold'em (RLCard)")

    # seed for reproducibility
    parser.add_argument('--seed', type=int, default=42)

    # training loop hyper-parameters
    parser.add_argument('--num_episodes', type=int, default=2_000_000)
    parser.add_argument('--num_eval_games', type=int, default=2_000)
    parser.add_argument('--evaluate_every', type=int, default=1_000)
    parser.add_argument('--opponent_update_every', type=int, default=10_000)

    # NFSP hyper-parameters
    parser.add_argument('--hidden_layers_sizes', type=int, nargs='+', default=[1024, 512, 1024, 512])
    parser.add_argument('--reservoir_buffer_capacity', type=int, default = 30_000_000)
    parser.add_argument('--anticipatory_param', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_every', type=int, default=256)
    parser.add_argument('--rl_learning_rate', type=float, default=0.1)
    parser.add_argument('--sl_learning_rate', type=float, default=0.01)
    parser.add_argument('--min_buffer_size_to_learn', type=int, default=10000)
    parser.add_argument('--q_replay_memory_size', type=int, default=600_000)
    parser.add_argument('--q_replay_memory_init_size', type=int, default=10_000)
    parser.add_argument('--q_update_target_estimator_every', type=int, default=1000)
    parser.add_argument('--q_discount_factor', type=float, default=0.99)
    parser.add_argument('--q_epsilon_start', type=float, default=0.08)
    parser.add_argument('--q_epsilon_end', type=float, default=0.0)
    parser.add_argument('--q_epsilon_decay_steps', type=int, default=int(5e6))
    parser.add_argument('--q_batch_size', type=int, default=256)
    parser.add_argument('--q_train_every', type=int, default=256)
    parser.add_argument('--q_mlp_layers', type=int, nargs='+', default=[1024, 512, 1024, 512])
    

    # misc
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--log_dir', type=str,
                        default=None)
    parser.add_argument('--verbose', type=bool, default=True)

    args = parser.parse_args()
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
