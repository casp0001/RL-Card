''' Evaluation panel: a fixed set of limit hold'em opponents used to monitor
training progress and to compare hyper-parameter configurations.

Members cover the tight/loose x passive/aggressive style space, so that every
kind of leak loses to somebody:

    random   uniform random legal action           sanity floor ("learned anything?")
    station  always check / call                   punishes bluffing too much
    maniac   always raise (else call / check)      punishes folding too much
    nit      tight-passive EquityAgent, no bluffs  punishes passivity
    tag      default EquityAgent                   the "plays reasonably" opponent
    lag      loose-aggressive EquityAgent          punishes over-folding to aggression
    honest   the old HonestAgent (not in the default panel)

Protocol:
  - duplicate deals: every deal is played twice with the seats swapped;
  - common random numbers: deal i of every evaluation uses seed
    EVAL_SEED + i for the env, numpy / random / torch and the members' own
    RNGs, so every checkpoint of every run faces the same cards and the same
    opponent "dice";
  - the global numpy / random / torch RNG states are saved and restored, so
    evaluating does not change the training run's random stream;
  - NFSP agents are evaluated with their average policy (restored afterwards).

Results are in mbb/hand from the evaluated agent's point of view. 'min' and
'mean' aggregate all members except 'random'. Use 'min' (the robustness
score) to select hyper-parameters.

Command line (final evaluation of a checkpoint with many deals):
    python eval_panel.py --kind dqn --ckpt experiments/<run> --deals 20000
    (kinds: dqn nfsp deepcfr ppo tag nit lag honest station maniac random)
'''

import os
import csv
import math
import time
import random
import argparse
import contextlib
from collections import Counter

import numpy as np
import torch

import rlcard_patches  # noqa: F401  (fixes rlcard limit hold'em bugs)
import rlcard
from rlcard.agents import RandomAgent

from equity_agent import EquityAgent, CALL, RAISE, FOLD, CHECK
from functions import ACTION_NAMES

EVAL_SEED = 10_000_000   # never used for training

# Style presets (calibrated with self-play style statistics, see the thesis
# notes): the nit folds ~25% as small blind and ~39% of big blinds facing a
# raise and never bluffs; the lag raises ~90% as small blind, 3-bets ~56% and
# bets ~60% when checked to.
NIT_PARAMS = {
    'raise_threshold': [0.58, 0.70, 0.72, 0.74],
    'reraise_step': [0.10, 0.08, 0.08, 0.08],
    'realisation': [0.60, 0.65, 0.75, 0.85],
    'bluff_frequency': [0.0, 0.0, 0.0, 0.0],
}
LAG_PARAMS = {
    'raise_threshold': [0.36, 0.52, 0.55, 0.58],
    'reraise_step': [0.08, 0.05, 0.05, 0.05],
    'realisation': [0.95, 1.0, 1.05, 1.1],
    'bluff_frequency': [0.0, 0.45, 0.40, 0.30],
    'bluff_max_equity': 0.45,
}


class FixedPolicyAgent(object):
    ''' Plays the first legal action from a preference list. '''
    use_raw = False

    def __init__(self, preference):
        self.preference = list(preference)

    def step(self, state):
        legal = state['legal_actions']
        for a in self.preference:
            if a in legal:
                return a
        return next(iter(legal))

    def eval_step(self, state):
        return self.step(state), {}


def _honest():
    from honest_agent_limited import HonestAgent
    return HonestAgent(num_actions=4)


MEMBERS = {
    'random': lambda: RandomAgent(num_actions=4),
    'station': lambda: FixedPolicyAgent([CHECK, CALL]),
    'maniac': lambda: FixedPolicyAgent([RAISE, CALL, CHECK]),
    'nit': lambda: EquityAgent(**NIT_PARAMS),
    'tag': lambda: EquityAgent(),
    'lag': lambda: EquityAgent(**LAG_PARAMS),
    'honest': _honest,
}
DEFAULT_MEMBERS = ('random', 'station', 'maniac', 'nit', 'tag', 'lag')
SANITY_MEMBERS = {'random'}   # excluded from 'min' / 'mean'


def parse_members(text):
    ''' 'random,tag' -> ('random', 'tag'); for argparse --eval_panel. '''
    names = tuple(n.strip() for n in text.split(',') if n.strip())
    unknown = [n for n in names if n not in MEMBERS]
    if unknown:
        raise ValueError(f'unknown panel members {unknown}; choose from {sorted(MEMBERS)}')
    return names


@contextlib.contextmanager
def preserved_global_rng():
    ''' Restore numpy / random / torch global RNG state on exit. '''
    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = torch.get_rng_state()
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        torch.set_rng_state(torch_state)


@contextlib.contextmanager
def eval_mode(agent, nfsp_policy='average_policy'):
    ''' Put an agent in the mode it should be evaluated in, restore after. '''
    if type(agent).__name__ == 'NFSPAgent':
        previous = agent.evaluate_with
        agent.evaluate_with = nfsp_policy
        try:
            yield
        finally:
            agent.evaluate_with = previous
    else:
        yield


class _ActionCounter(object):
    ''' Wraps the evaluated agent to record its action frequencies. '''

    def __init__(self, agent):
        self.agent = agent
        self.use_raw = getattr(agent, 'use_raw', False)
        self.counts = Counter()

    def eval_step(self, state):
        action, info = self.agent.eval_step(state)
        self.counts[int(action)] += 1
        return action, info

    def seed(self, seed):
        _reseed_agent(self.agent, seed)


def _reseed_agent(agent, seed):
    reseed = getattr(agent, 'seed', None)
    if callable(reseed):
        reseed(seed)


class Panel(object):
    ''' The evaluation panel.

    Args:
        members (iterable[str]): names from MEMBERS.
        num_deals (int): duplicate deals per member (2 hands each).
        eval_seed (int): seed of deal 0.
    '''

    def __init__(self, members=DEFAULT_MEMBERS, num_deals=1000, eval_seed=EVAL_SEED):
        self.members = {name: MEMBERS[name]() for name in members}
        self.num_deals = num_deals
        self.eval_seed = eval_seed
        self.env = rlcard.make('limit-holdem', config={'seed': eval_seed})

    def _reseed(self, seed, member, counted):
        self.env.seed(seed)
        np.random.seed(seed % (2 ** 32))
        random.seed(seed)
        torch.manual_seed(seed)
        _reseed_agent(member, seed + 1)
        counted.seed(seed + 2)

    def evaluate(self, agent, num_deals=None, nfsp_policy='average_policy'):
        ''' Evaluate `agent` (anything with an RLCard eval_step) against every
        member. Returns
            {'members': {name: {'mean': mbb/h, 'se': mbb/h}},
             'min': ..., 'mean': ..., 'actions': {action name: frequency},
             'seconds': ...}
        '''
        num_deals = num_deals or self.num_deals
        start = time.perf_counter()
        counted = _ActionCounter(agent)
        out = {}
        with preserved_global_rng(), eval_mode(agent, nfsp_policy):
            for name, member in self.members.items():
                per_deal = np.empty(num_deals)
                for i in range(num_deals):
                    seed = self.eval_seed + i
                    total = 0.0
                    for seat in (0, 1):
                        self._reseed(seed, member, counted)
                        self.env.set_agents([counted, member] if seat == 0 else [member, counted])
                        _, payoffs = self.env.run(is_training=False)
                        total += payoffs[seat]
                    per_deal[i] = total / 2
                mbb = per_deal * 1000   # payoffs are already in big blinds
                se = mbb.std(ddof=1) / math.sqrt(num_deals) if num_deals > 1 else float('nan')
                out[name] = {'mean': float(mbb.mean()), 'se': float(se)}
        scored = [v['mean'] for k, v in out.items() if k not in SANITY_MEMBERS]
        n_actions = sum(counted.counts.values()) or 1
        return {
            'members': out,
            'min': min(scored) if scored else float('nan'),
            'mean': float(np.mean(scored)) if scored else float('nan'),
            'actions': {ACTION_NAMES[a]: counted.counts[a] / n_actions for a in ACTION_NAMES},
            'seconds': time.perf_counter() - start,
        }


def format_results(results, prefix=''):
    members = '  '.join(f"{k} {v['mean']:+.0f}±{v['se']:.0f}" for k, v in results['members'].items())
    actions = ' '.join(f'{k}={v:.0%}' for k, v in results['actions'].items())
    return (f"{prefix}panel min {results['min']:+.1f}  mean {results['mean']:+.1f} mbb/h  |  {members}"
            f"\n{prefix}  actions: {actions}  ({results['seconds']:.0f}s)")


class PanelLogger(object):
    ''' Appends one row per evaluation to <log_dir>/panel.csv and redraws
    <log_dir>/panel.png. '''

    def __init__(self, log_dir, members, filename='panel'):
        os.makedirs(log_dir, exist_ok=True)
        self.members = list(members)
        self.csv_path = os.path.join(log_dir, f'{filename}.csv')
        self.fig_path = os.path.join(log_dir, f'{filename}.png')
        self.columns = (['episode', 'min', 'mean']
                        + [f'{m}_{s}' for m in self.members for s in ('mean', 'se')]
                        + [f'act_{a}' for a in ACTION_NAMES.values()])
        self.rows = []
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(self.columns)

    def log(self, episode, results):
        row = {'episode': episode, 'min': results['min'], 'mean': results['mean']}
        for m in self.members:
            row[f'{m}_mean'] = results['members'][m]['mean']
            row[f'{m}_se'] = results['members'][m]['se']
        for a, freq in results['actions'].items():
            row[f'act_{a}'] = freq
        self.rows.append(row)
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([row[c] for c in self.columns])
        self.plot()

    def plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        eps = [r['episode'] for r in self.rows]
        fig, ax = plt.subplots(figsize=(9, 5))
        for m in self.members:
            ax.plot(eps, [r[f'{m}_mean'] for r in self.rows], label=m)
        ax.plot(eps, [r['min'] for r in self.rows], 'k--', linewidth=2, label='min (excl. random)')
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.set_xlabel('episode / iteration')
        ax.set_ylabel('win rate (mbb/h)')
        ax.set_title('Evaluation panel')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.fig_path, dpi=110)
        plt.close(fig)


def add_panel_args(parser):
    ''' Shared --eval_deals / --eval_panel arguments for training scripts.
    Defaults are literals so train_gui.py can read them from the source. '''
    parser.add_argument('--eval_deals', type=int, default=1000,
                        help='Duplicate deals (2 hands each) per panel member per evaluation')
    parser.add_argument('--eval_panel', type=str, default='random,station,maniac,nit,tag,lag',
                        help=f'Comma-separated panel members, from: {",".join(MEMBERS)}')


def main():
    from agent_loading import load_policy, KINDS
    parser = argparse.ArgumentParser('Evaluate an agent against the panel')
    parser.add_argument('--kind', required=True, choices=KINDS)
    parser.add_argument('--ckpt', help='checkpoint file or run directory (dqn / nfsp / deepcfr / ppo)')
    parser.add_argument('--deals', type=int, default=20_000)
    parser.add_argument('--panel', type=str, default='random,station,maniac,nit,tag,lag')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    agent = load_policy(args.kind, args.ckpt, device=args.device)
    panel = Panel(parse_members(args.panel), num_deals=args.deals)
    print(format_results(panel.evaluate(agent)))


if __name__ == '__main__':
    main()
