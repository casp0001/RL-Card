
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


def compute_mbb_per_hand(avg_payoff, big_blind=2):
    return (avg_payoff / big_blind) * 1000

def fmt_actions(counts, total):
    if total == 0:
        return "no data"
    return (f"fold={counts.get(0,0)/total*100:.1f}% "
            f"call={counts.get(1,0)/total*100:.1f}% "
            f"raise={counts.get(2,0)/total*100:.1f}%")