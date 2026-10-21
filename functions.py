
# Applies fixes for rlcard 1.2.0 limit hold'em bugs (see the module).
import rlcard_patches  # noqa: F401

from rlcard.utils import (
    reorganize,
    set_seed,
    tournament,
    Logger,
    plot_curve,
)
import rlcard

from honest_agent_limited import HonestAgent
from rlcard.agents import NFSPAgent
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


def frozen_copy(agent):
    ''' Copy of an agent to use as a frozen self-play opponent.

    A plain copy.deepcopy also copies the replay / reservoir buffers (up to
    gigabytes, and seconds per snapshot) although the opponent never trains;
    here they are replaced by a NullBuffer that discards everything. A PPO
    opponent would also queue every move for an update that never comes, so
    its step() samples from the policy without recording anything. '''
    import copy
    import numpy as np
    from array_buffers import NullBuffer
    null = NullBuffer()
    memo = {}
    for owner, attr in ((agent, 'memory'), (agent, '_reservoir_buffer'),
                        (getattr(agent, '_rl_agent', None), 'memory')):
        buffer = getattr(owner, attr, None) if owner is not None else None
        if buffer is not None:
            memo[id(buffer)] = null
    opponent = copy.deepcopy(agent, memo)
    if type(agent).__name__ == 'PPOAgent':
        def step(state, policy=opponent):
            obs = np.asarray(state['obs'], dtype=np.float32)
            log_probs, _ = policy._policy_nograd(obs, list(state['legal_actions']))
            probs = np.exp(log_probs)
            return int(np.random.choice(len(probs), p=probs / probs.sum()))
        opponent.step = step
        opponent._pending = []
    return opponent

def plot_learning_curve(csv_path, fig_path, algorithm):
    ''' rlcard's plot_curve without leaving its figure open (a tuning worker
    plots once per trial and would otherwise collect figures), drawn with the
    non-interactive Agg backend unless a GUI backend is already in use. '''
    import sys
    import matplotlib
    if 'matplotlib.pyplot' not in sys.modules:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_curve(csv_path, fig_path, algorithm)
    plt.close('all')


def compute_mbb_per_hand(avg_payoff):
    # RLCard's hold'em payoffs are already divided by the big blind
    # (games/limitholdem/game.py get_payoffs), so avg_payoff is in BB/hand.
    return avg_payoff * 1000

# Action ids of RLCard's limit-holdem / leduc-holdem envs:
# env.actions == ['call', 'raise', 'fold', 'check']
ACTION_NAMES = {0: 'call', 1: 'raise', 2: 'fold', 3: 'check'}

def fmt_actions(counts, total):
    if total == 0:
        return "no data"
    return ' '.join(f"{name}={counts.get(a, 0)/total*100:.1f}%"
                    for a, name in ACTION_NAMES.items())

def str2bool(value):
    ''' argparse type for booleans: type=bool would turn "False" into True. '''
    return str(value).lower() in ('1', 'true', 'yes', 'y')