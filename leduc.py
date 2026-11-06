''' Exact exploitability on RLCard's Leduc Hold'em.

Leduc is small enough (3 ranks x 2 suits, one private and one public card,
two betting rounds) to walk the whole game tree, so here exploitability is
computed exactly instead of bounded from below (cf. exploitability.py's LBR
on limit hold'em).

The game is RLCard's 'leduc-holdem', which differs from the Leduc of the
literature (Southey et al., 2005):
  - blinds instead of antes: the small blind (random seat) posts 1 chip and
    acts first, the big blind posts 2; raises are 2 chips in round 1 and 4 in
    round 2, at most 2 raises per round;
  - fold is always legal (also when checking is possible);
  - round 2 starts with the player after the one who closed round 1;
  - payoffs are in big blinds (chips / 2), as in RLCard; 1 = 1000 mbb.
`LeducState` replays RLCard's rules (test_leduc.py checks it against the env)
and builds the agent's observation exactly as the env does (36 values: own
card, public card, own chips, opponent's chips). That observation does not
contain the betting sequence, so different histories can look the same to an
agent; the best response below has perfect recall regardless.

Exploitability of a policy that plays both seats (seats are random, so the
game is symmetric with value 0):
    expl(pi) = 1/2 * (BR value as small blind + BR value as big blind)
i.e. what a best response wins per hand against pi, averaged over seats
(NashConv / 2). It is 0 exactly for an equilibrium.

A policy here is a function `probs(info) -> 4 action probabilities` (action
ids as RLCard's: 0 call, 1 raise, 2 fold, 3 check) of an `Info`: the RLCard
state dict (`info.state`: 'obs', 'legal_actions', 'raw_obs',
'raw_legal_actions') plus the perfect-recall key (`info.key`: own rank,
public rank or None, betting history) for tabular policies.

    python leduc.py cfr --iterations 2000          # tabular CFR+, prints exploitability
    python leduc.py random                         # exploitability of the uniform policy
'''

import json
import time
import argparse
from collections import OrderedDict, namedtuple

import numpy as np

CALL, RAISE, FOLD, CHECK = 0, 1, 2, 3
ACTIONS = ['call', 'raise', 'fold', 'check']        # RLCard's leduc-holdem order
NUM_ACTIONS = 4
RANKS = (0, 1, 2)                                   # J, Q, K (RLCard's card2index)
RANK_NAMES = 'JQK'
SMALL_BLIND, BIG_BLIND = 1, 2
RAISE_AMOUNTS = (2, 4)
ALLOWED_RAISES = 2

Info = namedtuple('Info', 'state key')


class LeducState(object):
    ''' A betting state of RLCard's Leduc, seats by position: 0 = small blind
    (acts first), 1 = big blind. Mirrors rlcard.games.limitholdem.Round and
    rlcard.games.leducholdem.Game. Immutable: `child` returns a new state. '''
    __slots__ = ('round', 'pointer', 'raised', 'in_chips', 'have_raised', 'not_raised', 'folded', 'history')

    def __init__(self):
        self.round = 0
        self.pointer = 0
        self.raised = (SMALL_BLIND, BIG_BLIND)
        self.in_chips = (SMALL_BLIND, BIG_BLIND)
        self.have_raised = 0
        self.not_raised = 0
        self.folded = None
        self.history = ((), ())

    def legal_actions(self):
        legal = [CALL, RAISE, FOLD, CHECK]
        if self.have_raised >= ALLOWED_RAISES:
            legal.remove(RAISE)
        if self.raised[self.pointer] < max(self.raised):
            legal.remove(CHECK)
        if self.raised[self.pointer] == max(self.raised):
            legal.remove(CALL)
        return legal

    def is_terminal(self):
        return self.folded is not None or self.round >= 2

    def needs_public_card(self):
        ''' True right after round 1 ended (before round 2's first action). '''
        return self.round == 1 and not self.history[1]

    def child(self, action):
        s = LeducState.__new__(LeducState)
        p = self.pointer
        raised, chips = list(self.raised), list(self.in_chips)
        s.have_raised, s.not_raised, s.folded = self.have_raised, self.not_raised, None
        if action == CALL:
            chips[p] += max(raised) - raised[p]
            raised[p] = max(raised)
            s.not_raised += 1
        elif action == RAISE:
            amount = RAISE_AMOUNTS[self.round]
            chips[p] += max(raised) - raised[p] + amount
            raised[p] = max(raised) + amount
            s.have_raised += 1
            s.not_raised = 1
        elif action == FOLD:
            s.folded = p
        elif action == CHECK:
            s.not_raised += 1
        hist = list(self.history)
        hist[self.round] = hist[self.round] + (action,)
        s.history = tuple(hist)
        s.round, s.pointer = self.round, 1 - p
        if s.folded is None and s.not_raised >= 2:     # round over; round 2 starts with the next player
            s.round += 1
            raised = [0, 0]
            s.have_raised = s.not_raised = 0
        s.raised, s.in_chips = tuple(raised), tuple(chips)
        return s

    def payoff(self, seat, ranks, public):
        ''' Payoff of `seat` in big blinds; ranks = (small blind's, big blind's). '''
        other = 1 - seat
        if self.folded is not None:
            won = self.folded == other
        else:
            if ranks[seat] == public or (ranks[other] != public and ranks[seat] > ranks[other]):
                won = True
            elif ranks[other] == public or ranks[other] > ranks[seat]:
                won = False
            else:
                return 0.0
        return (self.in_chips[other] if won else -self.in_chips[seat]) / BIG_BLIND

    def info(self, rank, public):
        ''' The acting player's Info (RLCard state dict + perfect-recall key). '''
        p = self.pointer
        legal = self.legal_actions()
        obs = np.zeros(36)
        obs[rank] = 1
        if public is not None:
            obs[public + 3] = 1
        obs[self.in_chips[p] + 6] = 1
        obs[self.in_chips[1 - p] + 21] = 1
        state = {'obs': obs,
                 'legal_actions': OrderedDict((a, None) for a in legal),
                 'raw_obs': {'hand': rank, 'public_card': public, 'all_chips': list(self.in_chips),
                             'my_chips': self.in_chips[p], 'legal_actions': [ACTIONS[a] for a in legal]},
                 'raw_legal_actions': [ACTIONS[a] for a in legal]}
        return Info(state, (rank, public, self.history))


def _rank_weights(exclude=()):
    ''' Probability of each rank for the next card, given the dealt ranks. '''
    left = np.array([2.0 - sum(1 for e in exclude if e == r) for r in RANKS])
    return left / left.sum()


class _Table(object):
    ''' Caches a policy's probabilities by (observation, legal actions) for
    neural policies, or by perfect-recall key for tabular ones. '''

    def __init__(self, policy, by_key=False):
        self.policy, self.by_key, self.cache = policy, by_key, {}

    def __call__(self, state, rank, public):
        info = state.info(rank, public)
        k = info.key if self.by_key else (info.state['obs'].tobytes(), tuple(info.state['legal_actions']))
        p = self.cache.get(k)
        if p is None:
            p = np.asarray(self.policy(info), dtype=np.float64)
            legal = list(info.state['legal_actions'])
            mask = np.zeros(NUM_ACTIONS)
            mask[legal] = 1
            p = p * mask
            p = p / p.sum() if p.sum() > 0 else mask / mask.sum()
            self.cache[k] = p
        return p


def _br_value(state, seat, my_rank, public, w, table, strategy):
    ''' Best-response value of `seat` (holding my_rank) against the opponent's
    policy; w[r] = chance * opponent reach probability of each opponent rank.
    Returns sum_r w[r] * value. Records the BR's choices in `strategy`. '''
    if state.is_terminal():
        total = 0.0
        for r in RANKS:
            if w[r] > 0:
                ranks = (my_rank, r) if seat == 0 else (r, my_rank)
                total += w[r] * state.payoff(seat, ranks, public)
        return total
    if state.needs_public_card() and public is None:
        total = 0.0
        for pub in RANKS:
            w2 = np.array([w[r] * _rank_weights((my_rank, r))[pub] for r in RANKS])
            if w2.sum() > 0:
                total += _br_value(state, seat, my_rank, pub, w2, table, strategy)
        return total
    legal = state.legal_actions()
    if state.pointer == seat:
        values = {a: _br_value(state.child(a), seat, my_rank, public, w, table, strategy) for a in legal}
        best = max(legal, key=lambda a: values[a])
        strategy[(my_rank, public, state.history)] = best
        return values[best]
    total = 0.0
    probs = {r: table(state, r, public) for r in RANKS if w[r] > 0}
    for a in legal:
        w2 = np.array([w[r] * probs[r][a] if r in probs else 0.0 for r in RANKS])
        if w2.sum() > 0:
            total += _br_value(state.child(a), seat, my_rank, public, w2, table, strategy)
    return total


def best_response(policy, by_key=False):
    ''' Exact best response to `policy` (playing both seats).
    Returns {'exploitability', 'small_blind', 'big_blind'} in big blinds per
    hand, and 'strategy': {(seat, rank, public, history): action}. '''
    table = _Table(policy, by_key)
    out, strategy = {}, {}
    for seat, name in ((0, 'small_blind'), (1, 'big_blind')):
        value, strat = 0.0, {}
        for my_rank in RANKS:
            p_mine = _rank_weights()[my_rank]
            w = p_mine * _rank_weights((my_rank,))
            value += _br_value(LeducState(), seat, my_rank, None, w, table, strat)
        out[name] = value
        strategy.update({(seat,) + k: a for k, a in strat.items()})
    out['exploitability'] = (out['small_blind'] + out['big_blind']) / 2
    out['strategy'] = strategy
    return out


def exploitability(policy, by_key=False):
    return best_response(policy, by_key)['exploitability']


def _deals():
    ''' (sb rank, bb rank, public rank, probability) for every deal. '''
    for a in RANKS:
        for b in RANKS:
            for c in RANKS:
                p = _rank_weights()[a] * _rank_weights((a,))[b] * _rank_weights((a, b))[c]
                if p > 0:
                    yield a, b, c, p


def _match_value(state, ranks, public, tables):
    if state.is_terminal():
        return state.payoff(0, ranks, public)
    seat = state.pointer
    pub = public if state.round == 1 else None
    p = tables[seat](state, ranks[seat], pub)
    return sum(p[a] * _match_value(state.child(a), ranks, public, tables) for a in state.legal_actions() if p[a] > 0)


def match_value(policy_a, policy_b, a_by_key=False, b_by_key=False):
    ''' Exact expected payoff of policy_a against policy_b per hand (big
    blinds), averaged over seats. '''
    ta, tb = _Table(policy_a, a_by_key), _Table(policy_b, b_by_key)
    total = 0.0
    for sb, bb, pub, p in _deals():
        total += p * 0.5 * (_match_value(LeducState(), (sb, bb), pub, (ta, tb))
                            - _match_value(LeducState(), (sb, bb), pub, (tb, ta)))
    return total


def uniform_policy(info):
    legal = list(info.state['legal_actions'])
    p = np.zeros(NUM_ACTIONS)
    p[legal] = 1.0 / len(legal)
    return p


class TabularCFR(object):
    ''' CFR+ (Tammelin, 2014) with linearly weighted averaging over all deals
    and both seats, on perfect-recall information sets. The reference for
    "close to equilibrium" on this game. '''

    def __init__(self):
        self.regret, self.avg, self._delta = {}, {}, {}
        self.iterations = 0

    def _strategy(self, key, legal):
        r = self.regret.get(key)
        p = np.zeros(NUM_ACTIONS)
        if r is None or r[legal].sum() <= 0:
            p[legal] = 1.0 / len(legal)
        else:
            p[legal] = r[legal] / r[legal].sum()
        return p

    def _cfr(self, state, ranks, public, reach, chance, traverser):
        if state.is_terminal():
            return state.payoff(traverser, ranks, public)
        seat = state.pointer
        key = (ranks[seat], public if state.round == 1 else None, state.history)
        legal = state.legal_actions()
        sigma = self._strategy(key, legal)
        values = np.zeros(NUM_ACTIONS)
        # no pruning of the opponent's zero-probability actions: the average
        # strategy below them is weighted by the traverser's own reach only
        for a in legal:
            r = list(reach)
            r[seat] *= sigma[a]
            values[a] = self._cfr(state.child(a), ranks, public, r, chance, traverser)
        node = float(sigma @ values)
        if seat == traverser:
            delta = self._delta.setdefault(key, np.zeros(NUM_ACTIONS))
            delta[legal] += chance * reach[1 - seat] * (values[legal] - node)
            avg = self.avg.setdefault(key, np.zeros(NUM_ACTIONS))
            avg += self.iterations * chance * reach[seat] * sigma
        return node

    def iterate(self, n=1):
        for _ in range(n):
            self.iterations += 1
            for traverser in (0, 1):               # alternating updates
                self._delta = {}
                for sb, bb, pub, p in _deals():
                    self._cfr(LeducState(), (sb, bb), pub, [1.0, 1.0], p, traverser)
                for key, delta in self._delta.items():     # CFR+: floor once per pass
                    r = self.regret.setdefault(key, np.zeros(NUM_ACTIONS))
                    np.maximum(r + delta, 0.0, out=r)
        return self

    def average_policy(self):
        ''' The average strategy, by perfect-recall key (use by_key=True). '''
        avg = self.avg

        def policy(info):
            a = avg.get(info.key)
            legal = list(info.state['legal_actions'])
            p = np.zeros(NUM_ACTIONS)
            if a is None or a[legal].sum() <= 0:
                p[legal] = 1.0 / len(legal)
            else:
                p[legal] = a[legal] / a[legal].sum()
            return p
        return policy


def _action_frequencies(state, ranks, public, tables, seat, reach, freq):
    ''' Adds reach-weighted action probabilities of `seat` to freq. '''
    if state.is_terminal():
        return
    s = state.pointer
    pub = public if state.round == 1 else None
    p = tables[s](state, ranks[s], pub)
    if s == seat:
        freq += reach * p
    for a in state.legal_actions():
        if p[a] > 0:
            _action_frequencies(state.child(a), ranks, public, tables, seat, reach * p[a], freq)


_EQUILIBRIUM = {}


def equilibrium(iterations=1000):
    ''' CFR+ average strategy (~0.1 mbb/hand exploitable at 1000 iterations,
    15 s); computed once per process. '''
    if iterations not in _EQUILIBRIUM:
        _EQUILIBRIUM[iterations] = TabularCFR().iterate(iterations).average_policy()
    return _EQUILIBRIUM[iterations]


def policy_of(agent, nfsp_policy='average_policy'):
    ''' A training agent (or an agent_loading Policy) as a `probs(info)`
    function: the exact action distribution its eval_step plays (NFSP: the
    average policy, or the best-response Q-network's greedy policy). '''
    from agent_loading import Policy, DQNPolicy, NFSPPolicy, DeepCFRPolicy, PPOPolicy
    if isinstance(agent, Policy):
        return lambda info: agent.probs(info.state)
    name = type(agent).__name__
    if name == 'DeepCFREvalAgent':
        agent, name = agent._agent, 'DeepCFR'
    wrappers = {'DQNAgent': DQNPolicy, 'NFSPAgent': NFSPPolicy, 'DeepCFR': DeepCFRPolicy, 'PPOAgent': PPOPolicy}
    if name not in wrappers:
        raise ValueError(f'no Leduc policy for {name}')
    if name == 'NFSPAgent' and nfsp_policy == 'best_response':
        agent, name = agent._rl_agent, 'DQNAgent'
    previous = getattr(agent, 'evaluate_with', None)
    wrapped = wrappers[name](agent)
    if name == 'NFSPAgent':
        agent.evaluate_with = previous          # NFSPPolicy switches it to the average policy for good
    return lambda info: wrapped.probs_batch_obs(info.state['obs'][None].astype(np.float32),
                                                list(info.state['legal_actions']))[0]


class LeducPanel(object):
    ''' Drop-in for eval_panel.Panel on Leduc: exact numbers, no sampling.
    evaluate(agent) returns the Panel result format, in mbb/hand from the
    agent's point of view:
        members  best_response  -exploitability (what a best response wins, negated)
                 equilibrium    exact value against the CFR+ equilibrium (<= ~0)
                 random         exact value against the uniform policy
        min      = -exploitability (the score tuning maximises)
        mean     = value against the equilibrium
        actions  action frequencies, reach-weighted, in the match against the equilibrium
        exploitability, br_small_blind, br_big_blind (mbb/hand)
    NFSP agents are evaluated with their average policy, as by Panel. '''
    members = ('best_response', 'equilibrium', 'random')

    def __init__(self, cfr_iterations=1000):
        self.cfr_iterations = cfr_iterations

    def evaluate(self, agent, num_deals=None, nfsp_policy='average_policy'):
        start = time.perf_counter()
        eq = equilibrium(self.cfr_iterations)
        policy = policy_of(agent, nfsp_policy)
        br = best_response(policy)
        table, eq_table = _Table(policy), _Table(eq, True)
        freq = np.zeros(NUM_ACTIONS)
        for sb, bb, pub, p in _deals():
            _action_frequencies(LeducState(), (sb, bb), pub, (table, eq_table), 0, 0.5 * p, freq)
            _action_frequencies(LeducState(), (sb, bb), pub, (eq_table, table), 1, 0.5 * p, freq)
        freq /= freq.sum()
        expl = br['exploitability'] * 1000
        vs_eq = match_value(policy, eq, b_by_key=True) * 1000
        vs_random = match_value(policy, uniform_policy) * 1000
        members = {'best_response': -expl, 'equilibrium': vs_eq, 'random': vs_random}
        return {'members': {k: {'mean': v, 'se': 0.0} for k, v in members.items()},
                'min': -expl, 'mean': vs_eq, 'exploitability': expl,
                'br_small_blind': br['small_blind'] * 1000, 'br_big_blind': br['big_blind'] * 1000,
                'actions': {ACTIONS[a]: float(freq[a]) for a in range(NUM_ACTIONS)},
                'seconds': time.perf_counter() - start}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('evaluate', help='exact evaluation of a trained checkpoint')
    e.add_argument('--kind', required=True, choices=['dqn', 'nfsp', 'deepcfr', 'ppo'])
    e.add_argument('--ckpt', required=True)
    e.add_argument('--policy', default='average', choices=['average', 'best_response'],
                   help="NFSP only: the average policy (what NFSP plays) or the best-response Q-network")
    c = sub.add_parser('cfr')
    c.add_argument('--iterations', type=int, default=2000)
    c.add_argument('--report_every', type=int, default=100)
    sub.add_parser('random')
    a = p.parse_args()
    if a.command == 'evaluate':
        from agent_loading import load_policy, DQNPolicy
        policy = load_policy(a.kind, a.ckpt)
        if a.policy == 'best_response':
            policy = DQNPolicy(policy.agent._rl_agent)
        print(json.dumps(LeducPanel().evaluate(policy), indent=1))
        return
    if a.command == 'random':
        print(json.dumps({k: v for k, v in best_response(uniform_policy).items() if k != 'strategy'}))
        return
    cfr, t0 = TabularCFR(), time.time()
    while cfr.iterations < a.iterations:
        cfr.iterate(a.report_every)
        print(f'iteration {cfr.iterations:6d}  exploitability {exploitability(cfr.average_policy(), True) * 1000:9.2f} '
              f'mbb/hand  ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
