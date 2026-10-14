
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