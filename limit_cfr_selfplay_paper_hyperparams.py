import os
import argparse
import pickle

from honest_agent_limited import HonestAgent
import numpy as np
import rlcard
from functions import compute_mbb_per_hand, fmt_actions, str2bool
from rlcard.agents import CFRAgent
from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
)


# ---------------------------------------------------------------------------
# Action-counting wrapper
# ---------------------------------------------------------------------------

class CountingAgent:
    """
    Thin wrapper around a CFRAgent that intercepts eval_step calls so we can
    track action distributions without modifying the agent itself.
    """

    def __init__(self, agent):
        self.agent = agent
        self.counts = {0: 0, 1: 0, 2: 0}
        self.total = 0

    def reset(self):
        self.counts = {0: 0, 1: 0, 2: 0}
        self.total = 0

    def eval_step(self, state):
        action, info = self.agent.eval_step(state)
        self.counts[action] = self.counts.get(action, 0) + 1
        self.total += 1
        return action, info

    # Proxy everything else to the underlying agent so rlcard's tournament()
    # helper can call use_raw, etc.
    def __getattr__(self, name):
        return getattr(self.agent, name)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)

    # CFRAgent owns its environment internally — give it the training env.
    train_env = rlcard.make('limit-holdem', config={'seed': args.seed})
    eval_env  = rlcard.make('limit-holdem', config={'seed': args.seed})

    agent = CFRAgent(train_env, model_path=args.log_dir)

    # Optionally resume from a previous checkpoint.
    if args.resume:
        agent.load()
        print(f'Resumed from checkpoint at {args.log_dir} (iteration {agent.iteration})')

    honest_opponent = HonestAgent(num_actions=eval_env.num_actions)
    counting_wrapper = CountingAgent(agent)

    # Evaluation env: CFR average policy vs the honest baseline.
    eval_env.set_agents([counting_wrapper, honest_opponent])

    with Logger(args.log_dir) as logger:
        for iteration in range(1, args.num_iterations + 1):

            # -----------------------------------------------------------
            # One CFR iteration = full counterfactual-regret traversal for
            # every player, followed by a policy update.  No explicit
            # "opponent snapshot" concept is needed here — CFR maintains
            # its own average policy internally.
            # -----------------------------------------------------------
            agent.train()

            if args.verbose:
                print(f'\rIteration {iteration}/{args.num_iterations}', end='', flush=True)

            # -----------------------------------------------------------
            # Periodic evaluation
            # -----------------------------------------------------------
            if iteration % args.evaluate_every == 0:

                # --- average policy reward (main metric) ---------------
                counting_wrapper.reset()
                reward_avg = tournament(eval_env, args.num_eval_games)[0]
                action_counts_avg = dict(counting_wrapper.counts)
                total_steps_avg   = counting_wrapper.total

                # --- action distribution on a smaller sample -----------
                # (mirrors the NFSP script's 500-game action-count pass)
                counting_wrapper.reset()
                tournament(eval_env, 500)
                action_counts_sample = dict(counting_wrapper.counts)
                total_steps_sample   = counting_wrapper.total

                mbb = compute_mbb_per_hand(reward_avg)
                logger.log_performance(iteration, mbb)

                print(f'\n[Iteration {iteration}] mbb/h: {mbb:.2f}')
                print(f'  avg_policy reward : {reward_avg:.4f}')
                print(f'  avg_policy actions (eval  ): {fmt_actions(action_counts_avg,    total_steps_avg)}')
                print(f'  avg_policy actions (sample): {fmt_actions(action_counts_sample, total_steps_sample)}')
                print(f'  CFR iterations so far      : {agent.iteration}')

            # -----------------------------------------------------------
            # Periodic checkpoint
            # -----------------------------------------------------------
            if iteration % args.checkpoint_every == 0 and iteration > 0:
                ckpt_dir = os.path.join(args.log_dir, 'checkpoints', f'iteration_{iteration}')
                os.makedirs(ckpt_dir, exist_ok=True)
                # Temporarily redirect the agent's model_path to the checkpoint dir.
                original_path = agent.model_path
                agent.model_path = ckpt_dir
                agent.save()
                agent.model_path = original_path

                logger.csv_file.flush()
                plot_curve(logger.csv_path, logger.fig_path, 'cfr_limit_holdem')
                print(f'\nCheckpoint saved at iteration {iteration} → {ckpt_dir}')

        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Final save
    agent.save()
    print(f'\nModel saved to {args.log_dir}')

    plot_curve(csv_path, fig_path, 'cfr_limit_holdem')
    print(f'Learning curve saved to {fig_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CFR Self-Play — Limit Hold'em (RLCard)")

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)

    # Training loop
    parser.add_argument('--num_iterations',  type=int, default=10_000,
                        help='Number of CFR iterations (each traverses the full game tree)')
    parser.add_argument('--num_eval_games',  type=int, default=4_000,
                        help='Games played per evaluation checkpoint')
    parser.add_argument('--evaluate_every',  type=int, default=100,
                        help='Evaluate every N CFR iterations')
    parser.add_argument('--checkpoint_every', type=int, default=1_000,
                        help='Save a model checkpoint every N CFR iterations')

    # Misc
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for logs, curves, and saved model')
    parser.add_argument('--resume',  action='store_true',
                        help='Resume training from an existing checkpoint in --log_dir')
    parser.add_argument('--verbose', type=str2bool, default=True)

    args = parser.parse_args()

    # Auto-generate a descriptive log directory when none is provided.
    if args.log_dir is None:
        args.log_dir = (
            f'experiments/cfr/'
            f'iters{args.num_iterations}_'
            f'eval_every{args.evaluate_every}_'
            f'seed{args.seed}/'
        )
    os.makedirs(args.log_dir, exist_ok=True)

    train(args)
