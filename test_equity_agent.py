''' Tests for equity_agent.py.

Run:  python test_equity_agent.py            (about 5-10 minutes)
      python test_equity_agent.py --slow     (adds the exhaustive 7-card check,
                                              roughly another 10 minutes)
      python test_equity_agent.py -k river   (only tests whose name contains "river")
'''

import sys
import time
import math
import random
import argparse
import traceback
from itertools import combinations
from collections import Counter

import numpy as np
import rlcard

import equity_agent as ea
from equity_agent import (EquityAgent, evaluate, category, card_from_str, card_to_str,
                          hand_class, OpponentRange, equity_monte_carlo,
                          equity_river_exact, preflop_equity_table, CALL, RAISE, FOLD, CHECK)
from honest_agent_limited import HonestAgent, best_hand_rank
from rlcard.agents import RandomAgent


def cards(text):
    ''' 'As Kd 7h' -> [ints] (rank char then suit char, as poker players write). '''
    out = []
    for tok in text.split():
        out.append(card_from_str(tok[1].upper() + tok[0].upper()))
    return out


class SingleComboRange(OpponentRange):
    ''' A range holding exactly one hand (for exact match-up tests). '''
    def __init__(self, combo):
        self.fraction = None
        self.combos = [tuple(combo)]


# ---------------------------------------------------------------------------
# Hand evaluator
# ---------------------------------------------------------------------------

# Exact counts of 5-card hands (2,598,960) and of distinct hand values (7,462)
# per category, e.g. https://en.wikipedia.org/wiki/Poker_probability
FIVE_CARD_COUNTS = [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40]
FIVE_CARD_DISTINCT = [1277, 2860, 858, 858, 10, 1277, 156, 156, 10]
# Exact counts of best hands among all 133,784,560 7-card hands (same source).
SEVEN_CARD_COUNTS = [23294460, 58627800, 31433400, 6461620, 6180020,
                     4047644, 3473184, 224848, 41584]


def test_five_card_exhaustive():
    counts = [0] * 9
    distinct = [set() for _ in range(9)]
    for hand in combinations(range(52), 5):
        v = evaluate(hand)
        c = category(v)
        counts[c] += 1
        distinct[c].add(v)
    assert counts == FIVE_CARD_COUNTS, counts
    assert [len(d) for d in distinct] == FIVE_CARD_DISTINCT, [len(d) for d in distinct]
    assert sum(len(d) for d in distinct) == 7462


def test_seven_card_matches_best_of_21():
    ''' The 7-card shortcut must equal the best of its 21 five-card subsets. '''
    rng = random.Random(1)
    for _ in range(200_000):
        hand = rng.sample(range(52), 7)
        best = max(evaluate(c) for c in combinations(hand, 5))
        assert evaluate(hand) == best, [card_to_str(c) for c in hand]
    for n in (5, 6):
        for _ in range(50_000):
            hand = rng.sample(range(52), n)
            assert evaluate(hand) == max(evaluate(c) for c in combinations(hand, 5))


def test_seven_card_frequencies_sampled():
    ''' Category frequencies of 1M random 7-card hands vs the exact values. '''
    rng = random.Random(2)
    n = 1_000_000
    counts = Counter(category(evaluate(rng.sample(range(52), 7))) for _ in range(n))
    total = sum(SEVEN_CARD_COUNTS)
    for c in range(9):
        p = SEVEN_CARD_COUNTS[c] / total
        se = math.sqrt(p * (1 - p) / n)
        z = (counts[c] / n - p) / se
        assert abs(z) < 4.5, (ea.CATEGORY_NAMES[c], counts[c] / n, p, z)


def test_seven_card_exhaustive():   # slow
    counts = [0] * 9
    for hand in combinations(range(52), 7):
        counts[evaluate(hand) >> 20] += 1
    assert counts == SEVEN_CARD_COUNTS, counts


def test_ordering_matches_honest_agent_evaluator():
    ''' Independent implementation (honest_agent_limited.best_hand_rank, a
    tuple-based best-of-21 evaluator) must order random showdowns the same
    way. '''
    rng = random.Random(3)
    for _ in range(30_000):
        deal = rng.sample(range(52), 9)
        board, h1, h2 = deal[:5], deal[5:7], deal[7:]
        mine = (evaluate(h1 + board) > evaluate(h2 + board)) - (evaluate(h1 + board) < evaluate(h2 + board))
        s = lambda cs: [card_to_str(c) for c in cs]
        r1, r2 = best_hand_rank(s(h1), s(board)), best_hand_rank(s(h2), s(board))
        theirs = (r1 > r2) - (r1 < r2)
        assert mine == theirs, (s(board), s(h1), s(h2))


def test_specific_hands():
    v = lambda t: evaluate(cards(t))
    cat = lambda t: ea.CATEGORY_NAMES[category(v(t))]
    assert cat('Ah 2d 3c 4s 5h 9d Kc') == 'straight'
    assert v('Ah 2d 3c 4s 5h 9d Kc') < v('2d 3c 4s 5h 6d 9d Kc')       # wheel is the lowest straight
    assert v('Ts Js Qs Ks As 2d 3c') > v('9s Ts Js Qs Ks 2d 3c')       # royal > king-high SF
    assert cat('Ah 2h 3h 4h 5h Kd Kc') == 'straight flush'              # steel wheel
    assert cat('Kh Qh 9h 4h 2h Kd Kc') == 'flush'                       # flush beats trips
    assert cat('Kh Kd Kc 9h 9d 9c 2s') == 'full house'                  # two trips -> full house
    assert v('Kh Kd Kc 9h 9d 9c 2s') > v('Kh Kd Kc 8h 8d Ac 2s')        # KKK99 > KKK88
    assert v('Ah Ad 7c 7d 5h 5s Kc') > v('Ah Ad 7c 7d 5h 5s Qc')        # kicker after two pair
    assert v('Ah Ad 7c 7d 5h 5s 2c') == v('Ah Ad 7c 7d 5h 4s 2c')       # 3rd pair only a kicker
    assert v('Ah Ad 7c 7d 6h 6s 2c') > v('Ah Ad 7c 7d 5h 5s 2c')        # kicker 6 beats kicker 5
    assert v('9h 9d 9c 9s Ah 2d 3c') > v('9h 9d 9c 9s Kh Kd Kc')        # quads kicker
    assert cat('2h 3d 4c 5s 7h 8d 9c') == 'high card'
    assert v('Ah Kd 9c 7s 5h 3d 2c') > v('Ah Kd 9c 7s 4h 3d 2c')        # 5th kicker counts
    assert v('Ah Kd 9c 7s 5h 3d 2c') == v('Ah Kd 9c 7s 5h 4d 2c')       # 6th card does not
    # board plays: both players share the same best five
    board = cards('Ts Js Qs Ks As')
    assert evaluate(cards('2h 3d') + board) == evaluate(cards('4h 5d') + board)


def test_card_conversion_roundtrip():
    for c in range(52):
        assert card_from_str(card_to_str(c)) == c
    assert hand_class(*cards('Ah Kh')) == 'AKs'
    assert hand_class(*cards('7d 2c')) == '72o'
    assert hand_class(*cards('2c 7d')) == '72o'
    assert hand_class(*cards('9s 9d')) == '99'
    assert len(ea.ALL_CLASSES) == 169 and len(ea.ALL_COMBOS) == 1326


# ---------------------------------------------------------------------------
# Equity
# ---------------------------------------------------------------------------

# Widely published heads-up all-in equities vs a random hand (PokerStove-style
# enumeration; AA 85.2% also e.g. https://poker.stephan-brumme.com/AA.html).
REFERENCE_PREFLOP = {'AA': 0.852, 'KK': 0.824, 'QQ': 0.799, 'AKs': 0.670,
                     'AKo': 0.653, '72o': 0.346, '32o': 0.323}


def test_preflop_table():
    t = preflop_equity_table()
    assert set(t) == set(ea.ALL_CLASSES)
    assert max(t, key=t.get) == 'AA'
    pairs = [r * 2 for r in reversed(ea.RANKS)]
    assert all(t[a] > t[b] for a, b in zip(pairs, pairs[1:])), 'pairs not monotone'
    for hi, lo in combinations(reversed(ea.RANKS), 2):
        assert t[hi + lo + 's'] > t[hi + lo + 'o'], hi + lo
    for cls, ref in REFERENCE_PREFLOP.items():
        assert abs(t[cls] - ref) < 0.006, (cls, t[cls], ref)


def exact_preflop(hero, villain):
    ''' Exact all-in equity by enumerating all 1,712,304 boards. '''
    rest = [c for c in range(52) if c not in hero and c not in villain]
    total = 0.0
    n = 0
    for board in combinations(rest, 5):
        board = list(board)
        h, v = evaluate(hero + board), evaluate(villain + board)
        total += 1.0 if h > v else (0.5 if h == v else 0.0)
        n += 1
    assert n == 1_712_304
    return total / n


def test_known_matchups():
    # AA vs KK all-in pre-flop: 81.9% for the aces averaged over suits (e.g.
    # DeucesCracked's exact pre-flop match-up data). Per AA combo the 6 KK
    # combos share 2 suits (1 combo), 1 suit (4) or no suit (1) with it.
    aa = cards('Ah As')
    same, one, none = (exact_preflop(aa, cards(k)) for k in ('Kh Ks', 'Kh Kd', 'Kd Kc'))
    average = (same + 4 * one + none) / 6
    assert same > one > none, (same, one, none)       # shared suits block KK's flushes
    assert abs(average - 0.819) < 0.002, average
    # The Monte Carlo estimator agrees with exact enumeration.
    rng = random.Random(4)
    mc = equity_monte_carlo(aa, [], SingleComboRange(cards('Kd Kc')), rng, 60_000)
    assert abs(mc - none) < 4 * math.sqrt(none * (1 - none) / 60_000), (mc, none)
    # Coin flip: 22 vs AKo is ~52-53% for the pair.
    eq = equity_monte_carlo(cards('2h 2d'), [], SingleComboRange(cards('As Kc')), rng, 60_000)
    assert 0.505 < eq < 0.545, eq


def test_river_exact_symmetry_and_extremes():
    board = cards('2c 7d 9h Js Kd')
    a, b = cards('Ah Kh'), cards('9c 9s')
    ab = equity_river_exact(a, board, SingleComboRange(b))
    ba = equity_river_exact(b, board, SingleComboRange(a))
    assert ab + ba == 1.0 and ba == 1.0          # set of nines beats a pair of kings
    full = OpponentRange(1.0)
    nuts = equity_river_exact(cards('Qs Td'), board, full)         # broadway straight
    assert nuts > 0.99, nuts
    tie_board = cards('Ts Js Qs Ks As')                            # royal on board
    assert equity_river_exact(cards('2h 3d'), tie_board, full) == 0.5


def test_monte_carlo_matches_exact_turn():
    ''' MC flop/turn equity agrees with exact enumeration of the river. '''
    hero, board = cards('Ah Th'), cards('Kh 7h 2c 9d')      # nut flush draw on the turn
    full = OpponentRange(1.0)
    dead = set(hero) | set(board)
    rivers = [c for c in range(52) if c not in dead]
    # River uniform over the unseen cards, then the opponent uniform over the
    # combos that avoid it: the same joint distribution the MC samples from.
    exact = sum(equity_river_exact(hero, board + [r], full) for r in rivers) / len(rivers)
    mc = equity_monte_carlo(hero, board, full, random.Random(5), 40_000)
    assert abs(mc - exact) < 0.012, (mc, exact)


def test_range_narrowing():
    agent = EquityAgent(seed=0)
    t = preflop_equity_table()
    k7 = cards('Kd 7c')
    assert abs(agent.equity(k7, [], 0) - t['K7o']) < 1e-12        # table lookup
    narrowed = [agent.equity(k7, [], n) for n in range(4)]
    assert narrowed[0] > narrowed[1] > narrowed[2] > narrowed[3], narrowed
    # a tight range is still crushed by aces
    assert agent.equity(cards('Ah As'), [], 3) > 0.75
    assert len(OpponentRange(0.4).combos) == round(0.4 * 1326)


# ---------------------------------------------------------------------------
# Agent behaviour
# ---------------------------------------------------------------------------

def raw(hand, board, all_chips, my_chips, legal, raise_nums):
    return {'hand': hand, 'public_cards': board, 'all_chips': all_chips,
            'my_chips': my_chips, 'legal_actions': legal, 'raise_nums': raise_nums}


def test_rlcard_patches():
    ''' rlcard_patches.py: no stale raise counts at the start of a hand, and
    step_back restores them. '''
    env = rlcard.make('limit-holdem', config={'seed': 0, 'allow_step_back': True})
    for _ in range(50):
        state, _ = env.reset()
        assert state['raw_obs']['raise_nums'] == [0, 0, 0, 0]
        # one-hot "0 raises" bit set for each of the 4 rounds
        assert state['obs'][52:].nonzero()[0].tolist() == [0, 5, 10, 15], state['obs'][52:]
        while not env.is_over():   # raise whenever possible
            legal = state['legal_actions']
            state, _ = env.step(RAISE if RAISE in legal else (CALL if CALL in legal else CHECK))
    # a raise followed by step_back must restore the raise count
    state, _ = env.reset()
    before = list(state['raw_obs']['raise_nums'])
    env.step(RAISE)
    restored, _ = env.step_back()
    assert restored['raw_obs']['raise_nums'] == before, restored['raw_obs']['raise_nums']


def test_opponent_raise_counting():
    f = EquityAgent._opponent_raises
    assert f([]) == 0
    assert f([(0, 'raise')]) == 1                          # opponent (seat 0) raised
    assert f([(1, 'raise'), (0, 'raise'), (1, 'raise'), (0, 'call')]) == 1
    assert f([(0, 'raise'), (1, 'call'), (1, 'check'), (0, 'raise')]) == 2


def test_unknown_param_rejected():
    try:
        EquityAgent(raise_treshold=[0.5] * 4)
    except ValueError:
        return
    raise AssertionError('typo in parameter name was accepted')


def test_spot_decisions():
    agent = EquityAgent(seed=0)
    n = 400
    # River, nut straight, facing a bet: never folds, nearly always raises
    obs = raw(['SQ', 'DT'], ['C2', 'D7', 'H9', 'SJ', 'DK'], [8, 12], 8, ['call', 'raise', 'fold'], [1, 1, 0, 1])
    acts = Counter(agent.decide(obs, [(1, 'raise')])[0] for _ in range(n))
    assert acts[FOLD] == 0 and acts[RAISE] > 0.9 * n, acts
    # River, 3-high nothing, facing a bet with ~4:1 pot odds: folds nearly always
    obs = raw(['C3', 'H4'], ['C2', 'D7', 'H9', 'SJ', 'DK'], [8, 12], 8, ['call', 'raise', 'fold'], [1, 1, 0, 1])
    acts = Counter(agent.decide(obs, [(1, 'raise')])[0] for _ in range(n))
    assert acts[FOLD] > 0.95 * n, acts
    # Same trash when checked to on the river: never folds, bluffs ~15%
    obs = raw(['C3', 'H4'], ['C2', 'D7', 'H9', 'SJ', 'DK'], [8, 8], 8, ['raise', 'fold', 'check'], [1, 1, 0, 0])
    acts = Counter(agent.decide(obs, [(1, 'check')])[0] for _ in range(4000))
    assert acts[FOLD] == 0, acts
    assert abs(acts[RAISE] / 4000 - 0.15) < 0.025, acts
    # Pre-flop small blind with aces: raises
    obs = raw(['SA', 'HA'], [], [1, 2], 1, ['call', 'raise', 'fold'], [0, 0, 0, 0])
    acts = Counter(agent.decide(obs, [])[0] for _ in range(n))
    assert acts[RAISE] == n, acts
    # Pre-flop small blind with 72o: completing costs 1 into 3 (25% pot odds),
    # 72o (~35% equity) vs 31% required: mostly continues, mostly by
    # completing; the soft thresholds mix in some folds and some raises
    obs = raw(['S7', 'H2'], [], [1, 2], 1, ['call', 'raise', 'fold'], [0, 0, 0, 0])
    acts = Counter(agent.decide(obs, [])[0] for _ in range(4000))
    assert acts[CALL] > acts[FOLD] > 0 and acts[CALL] > acts[RAISE] > 0, acts
    assert acts[FOLD] < 0.35 * 4000 and acts[RAISE] < 0.25 * 4000, acts


class _Checked(object):
    ''' Wraps an agent and asserts every action is legal and never a fold
    when checking was free. '''
    use_raw = False

    def __init__(self, agent):
        self.agent = agent
        self.decisions = 0

    def _check(self, state, action):
        legal = state['legal_actions']
        assert action in legal, (action, list(legal))
        if CHECK in legal:
            assert action != FOLD, 'folded when check was free'
        self.decisions += 1

    def step(self, state):
        a = self.agent.step(state)
        self._check(state, a)
        return a

    def eval_step(self, state):
        a, info = self.agent.eval_step(state)
        self._check(state, a)
        return a, info


def test_always_legal_and_never_folds_for_free():
    env = rlcard.make('limit-holdem', config={'seed': 6})
    for opponent in (RandomAgent(4), EquityAgent(seed=1)):
        checked = _Checked(EquityAgent(seed=0))
        for seat in (0, 1):
            env.set_agents([checked, opponent] if seat == 0 else [opponent, checked])
            for _ in range(1500):
                env.run(is_training=False)
        assert checked.decisions > 3000


def test_deterministic_given_seeds():
    def play():
        env = rlcard.make('limit-holdem', config={'seed': 7})
        env.set_agents([EquityAgent(seed=3), EquityAgent(seed=4)])
        return [tuple(env.run(is_training=False)[1]) for _ in range(300)]
    assert play() == play()


# ---------------------------------------------------------------------------
# Matches (duplicate format: every deal is played twice with seats swapped)
# ---------------------------------------------------------------------------

class Fixed(object):
    ''' First legal action from a preference list. '''
    use_raw = False

    def __init__(self, order):
        self.order = order

    def step(self, state):
        for a in self.order:
            if a in state['legal_actions']:
                return a

    def eval_step(self, state):
        return self.step(state), {}


def duplicate_match(agent, opponent, deals, seed=1000):
    ''' Mean and standard error (mbb/hand) of `agent` vs `opponent`. '''
    env = rlcard.make('limit-holdem', config={'seed': seed})
    results = []
    for i in range(deals):
        env.seed(seed + i)
        env.set_agents([agent, opponent])
        p0 = env.run(is_training=False)[1][0]
        env.seed(seed + i)
        env.set_agents([opponent, agent])
        p1 = env.run(is_training=False)[1][1]
        results.append((p0 + p1) / 2)
    r = np.array(results) * 1000   # payoffs are already in big blinds
    return r.mean(), r.std(ddof=1) / math.sqrt(len(r))


def test_matches():
    deals = 10_000
    opponents = {
        'random': RandomAgent(4),
        'calling station': Fixed([CHECK, CALL]),
        'maniac (always raise)': Fixed([RAISE, CALL, CHECK]),
        'honest (old rule bot)': HonestAgent(4),
    }
    for name, opp in opponents.items():
        mean, se = duplicate_match(EquityAgent(seed=11), opp, deals)
        print(f'    vs {name:24s}: {mean:+7.1f} ± {se:5.1f} mbb/h')
        assert mean > 4 * se and mean > 50, (name, mean, se)
    mean, se = duplicate_match(EquityAgent(seed=11), EquityAgent(seed=12), deals)
    print(f'    vs {"itself":24s}: {mean:+7.1f} ± {se:5.1f} mbb/h')
    assert abs(mean) < 4 * se + 5, (mean, se)


def test_speed():
    env = rlcard.make('limit-holdem', config={'seed': 8})
    agent = EquityAgent(seed=0)
    timed = _Checked(agent)
    env.set_agents([timed, EquityAgent(seed=1)])
    t = time.perf_counter()
    for _ in range(1500):
        env.run(is_training=False)
    per_decision = (time.perf_counter() - t) / timed.decisions / 2 * 1000
    print(f'    ~{per_decision:.2f} ms per decision (cold cache)')
    assert per_decision < 5, per_decision


# ---------------------------------------------------------------------------

SLOW = {'test_seven_card_exhaustive'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slow', action='store_true')
    parser.add_argument('-k', default='')
    args = parser.parse_args()
    tests = [(n, f) for n, f in list(globals().items())
             if n.startswith('test_') and callable(f) and args.k in n
             and (args.slow or n not in SLOW)]
    failed = 0
    for name, fn in tests:
        t = time.perf_counter()
        try:
            fn()
            print(f'PASS {name} ({time.perf_counter() - t:.1f}s)', flush=True)
        except Exception:
            failed += 1
            print(f'FAIL {name} ({time.perf_counter() - t:.1f}s)', flush=True)
            traceback.print_exc()
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
