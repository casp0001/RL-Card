import os
import argparse
import copy
import contextlib


import rlcard
from functions import fmt_actions, str2bool
from eval_panel import Panel, PanelLogger, add_panel_args, parse_members, format_results
from ppo_agent import PPOAgent

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

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Initialise the main PPO agent
    # Limit Hold'em: state_shape is [72], num_actions is 4 (call/raise/fold/check)
    agent = PPOAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=args.mlp_layers,
        activation=args.activation,
        learning_rate=args.lr,
        discount_factor=args.discount_factor,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        value_clip_range=args.value_clip_range,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        update_every=args.update_every,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        reward_scaling=args.reward_scaling,
        target_kl=args.target_kl,
        evaluate_with=args.evaluate_with,
        device=args.device,
    )

    # Opponent pool for self-play: starts as a frozen copy of the initial agent
    opponent = copy.deepcopy(agent)

    # Evaluation panel of fixed opponents (see eval_panel.py)
    panel = Panel(parse_members(args.eval_panel), num_deals=args.eval_deals)
    panel_logger = PanelLogger(args.log_dir, panel.members)

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

                    # Run one episode and feed transitions to the learning agent.
                    # Only the learner's own trajectory is fed: the snapshot
                    # opponent is a different policy, so its transitions are
                    # off-policy and would break the PPO importance ratio.
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
                print(format_results(results, prefix=f'[Episode {episode}] '))
                if agent.last_loss is not None:
                    print(
                        f'  updates: {agent.train_t} | '
                        f'policy: {agent.last_loss["policy_loss"]:.5f} | '
                        f'value: {agent.last_loss["value_loss"]:.5f} | '
                        f'entropy: {agent.last_loss["entropy"]:.4f} | '
                        f'approx_kl: {agent.last_loss["approx_kl"]:.5f}'
                    )

            # Periodic checkpoints
            if args.checkpoint_every > 0 and episode % args.checkpoint_every == 0 and episode > 0:
                ckpt_dir = os.path.join(args.log_dir, 'checkpoints', f'episode_{episode}')
                os.makedirs(ckpt_dir, exist_ok=True)
                agent.save_checkpoint(ckpt_dir)
                print(f'\nCheckpoint saved at episode {episode} to {ckpt_dir}')

        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Save final model
    agent.save_checkpoint(args.log_dir)
    print(f'\nModel saved to {args.log_dir}')

    # Plot learning curve
    plot_curve(csv_path, fig_path, 'ppo_selfplay_limit_holdem')
    print(f'Learning curve saved to {fig_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PPO Self-Play — Limit Hold'em (RLCard)")

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)

    # Training loop
    parser.add_argument('--num_episodes', type=int, default=200_000)
    add_panel_args(parser)
    parser.add_argument('--evaluate_every', type=int, default=2_000,
                        help='Evaluate against the panel every N episodes')
    parser.add_argument('--opponent_update_every', type=int, default=2_000,
                        help='Freeze a new opponent snapshot every N episodes')
    parser.add_argument('--checkpoint_every', type=int, default=0,
                        help='Save an intermediate checkpoint every N episodes (0 = never)')

    # PPO hyper-parameters
    parser.add_argument('--mlp_layers', type=int, nargs='+', default=[128, 128],
                        help='Hidden layer sizes of the actor-critic trunk, e.g. --mlp_layers 256 256')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'tanh'],
                        help='Activation function for the MLP trunk')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--value_clip_range', type=float, default=None,
                        help='Clip the value loss around the old prediction (None = off)')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy bonus; keeps the poker policy mixed rather than exploitable')
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--update_every', type=int, default=2048,
                        help='Transitions collected per PPO update (~500 hands at 4 actions/hand)')
    parser.add_argument('--num_epochs', type=int, default=4,
                        help='Passes over the rollout buffer per update')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Minibatch size within an epoch')
    parser.add_argument('--reward_scaling', type=float, default=1.0,
                        help='Internal reward multiplier; rescales value targets only')
    parser.add_argument('--target_kl', type=float, default=None,
                        help='Stop the epoch loop early past this approximate KL (None = off)')
    parser.add_argument('--evaluate_with', type=str, default='stochastic',
                        choices=['stochastic', 'greedy'])

    # Misc
    parser.add_argument('--device', type=str, default=None,
                        help='torch device, e.g. "cpu" or "cuda:0" (None = auto)')
    parser.add_argument('--log_dir', type=str,
                        default=None)
    parser.add_argument('--verbose', type=str2bool, default=True)

    args = parser.parse_args()
    # log directory dynamic creation
    if args.log_dir is None:
        layers_str = '_'.join(str(x) for x in args.mlp_layers)
        args.log_dir = (
            f'experiments/limit_holdem_ppo_selfplay/'
            f'activation{args.activation}_'
            f'lr{args.lr}_upd{args.update_every}_ep{args.num_epochs}_bs{args.batch_size}_'
            f'layers{layers_str}_clip{args.clip_range}_ent{args.entropy_coef}_'
            f'gamma{args.discount_factor}_lam{args.gae_lambda}_'
            f'opp{args.opponent_update_every}_seed{args.seed}/'
        )
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)
