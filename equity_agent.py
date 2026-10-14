''' Equity / pot-odds rule-based agent for RLCard limit hold'em.

A "reasonable" hand-crafted opponent in the spirit of the University of
Alberta's Loki/Poki bots (Billings et al., 2002, "The challenge of poker"):

1. Hand strength is the probability of winning at showdown (equity) against
   the opponent's estimated range:
     - pre-flop: a precomputed table of all 169 starting-hand classes versus a
       random hand (preflop_equity.json, built by `python equity_agent.py
       --build-preflop`), or a Monte Carlo estimate once the opponent has
       raised and its range is narrower;
     - flop / turn: Monte Carlo roll-outs of the opponent's hand and the rest
       of the board. Equity over the remaining board already credits draws, so
       a separate hand-potential term (Billings' EHS) is not needed;
     - river: exact enumeration of every opponent hand in its range.
2. The opponent's range narrows with its aggression: after its n-th raise in
   the hand only the top `range_fractions[n]` of starting hands (ranked by
   pre-flop equity) are considered.
3. Decisions compare equity with thresholds:
     - raise when equity clears a street-dependent value threshold, which
       rises with every raise already made this round;
     - facing a bet, continue when equity beats the pot odds (scaled by an
       equity-realisation factor on earlier streets);
     - never fold when checking is free;
     - bluff the weakest hands at a small, street-dependent frequency when
       checked to.
   Every threshold is a logistic "soft" threshold, so the strategy is mixed
   rather than a deterministic lookup table a learner could trivially exploit.

All randomness (roll-outs and mixing) comes from the agent's own seeded RNG, so
results are reproducible and the agent does not disturb numpy's global RNG
used by the other agents.
'''

import os
import json
import math
import random
import zlib
import argparse
from itertools import combinations

# raw_obs['raise_nums'] is wrong at the start of a hand without this.
import rlcard_patches  # noqa: F401


# ---------------------------------------------------------------------------
# Cards
#
# A card is an int 0..51: rank = card >> 2 (0 = deuce .. 12 = ace),
# suit = card & 3. RLCard strings are suit + rank, e.g. 'SA', 'HT', 'D2'.
# ---------------------------------------------------------------------------

RANKS = '23456789TJQKA'
SUITS = 'SHDC'


def card_from_str(card_str):
    return (RANKS.index(card_str[1]) << 2) | SUITS.index(card_str[0])


def card_to_str(card):
    return SUITS[card & 3] + RANKS[card >> 2]


FULL_DECK = list(range(52))


# ---------------------------------------------------------------------------
# Hand evaluator (5, 6 or 7 cards)
#
# Returns an int; a larger value is a better hand and equal values tie.
# Layout: category << 20 | five 4-bit tie-break ranks (most significant first).
# ---------------------------------------------------------------------------

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = ['high card', 'pair', 'two pair', 'three of a kind', 'straight',
                  'flush', 'full house', 'four of a kind', 'straight flush']


def _pack(category, ranks):
    value = category
    for i in range(5):
        value = (value << 4) | (ranks[i] if i < len(ranks) else 0)
    return value


def _straight_high(mask):
    ''' Highest straight in a 13-bit rank mask, or -1. Ace also plays low. '''
    # Shift ranks up one bit so bit 0 can hold the ace again as the low card:
    # bit 0 = ace-low, bits 1..13 = deuce..ace.
    mask = (mask << 1) | ((mask >> 12) & 1)
    for high in range(13, 3, -1):
        if (mask >> (high - 4)) & 0b11111 == 0b11111:
            return high - 1   # back to 0..12 ranks (3 = five-high wheel)
    return -1


# Lookup tables over 13-bit rank masks
_STRAIGHT_HIGH = [_straight_high(m) for m in range(1 << 13)]
_BITS_DESC = [[r for r in range(12, -1, -1) if m >> r & 1] for m in range(1 << 13)]


def evaluate(cards):
    ''' Value of the best 5-card hand that can be made from `cards` (5-7 ints). '''
    counts = [0] * 13
    suit_masks = [0, 0, 0, 0]
    rank_mask = 0
    for c in cards:
        r = c >> 2
        counts[r] += 1
        suit_masks[c & 3] |= 1 << r
        rank_mask |= 1 << r

    # With at most 7 cards a flush excludes quads and full houses, so a flush
    # (or straight flush) can be returned straight away.
    for m in suit_masks:
        bits = _BITS_DESC[m]
        if len(bits) >= 5:
            high = _STRAIGHT_HIGH[m]
            if high >= 0:
                return _pack(STRAIGHT_FLUSH, [high])
            return _pack(FLUSH, bits[:5])

    quads, trips, pairs, singles = [], [], [], []
    for r in range(12, -1, -1):
        n = counts[r]
        if n == 4:
            quads.append(r)
        elif n == 3:
            trips.append(r)
        elif n == 2:
            pairs.append(r)
        elif n == 1:
            singles.append(r)

    if quads:
        kicker = max(trips + pairs + singles)
        return _pack(QUADS, [quads[0], kicker])
    if trips and (len(trips) > 1 or pairs):
        # second trips can serve as the pair of a full house
        pair = max(trips[1:] + pairs)
        return _pack(FULL_HOUSE, [trips[0], pair])
    high = _STRAIGHT_HIGH[rank_mask]
    if high >= 0:
        return _pack(STRAIGHT, [high])
    if trips:
        return _pack(TRIPS, [trips[0]] + singles[:2])
    if len(pairs) >= 2:
        kicker = max(pairs[2:] + singles)
        return _pack(TWO_PAIR, [pairs[0], pairs[1], kicker])
    if pairs:
        return _pack(PAIR, [pairs[0]] + singles[:3])
    return _pack(HIGH_CARD, singles[:5])


def category(value):
    return value >> 20


# ---------------------------------------------------------------------------
# Starting hands
# ---------------------------------------------------------------------------

def hand_class(c1, c2):
    ''' 169-class label of two hole cards, e.g. 'AA', 'AKs', '72o'. '''
    r1, r2 = c1 >> 2, c2 >> 2
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        return RANKS[r1] * 2
    return RANKS[r1] + RANKS[r2] + ('s' if (c1 & 3) == (c2 & 3) else 'o')


ALL_COMBOS = list(combinations(FULL_DECK, 2))          # 1326 hole-card combos
ALL_CLASSES = sorted({hand_class(a, b) for a, b in ALL_COMBOS})   # 169 classes

PREFLOP_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'preflop_equity.json')


def build_preflop_table(samples_per_class=60_000, seed=0):
    ''' Monte Carlo equity of each of the 169 classes against a random hand. '''
    rng = random.Random(seed)
    table = {}
    by_class = {}
    for a, b in ALL_COMBOS:
        by_class.setdefault(hand_class(a, b), []).append((a, b))
    for i, cls in enumerate(ALL_CLASSES):
        total = 0.0
        combos = by_class[cls]
        for s in range(samples_per_class):
            hero = combos[s % len(combos)]
            deck = [c for c in FULL_DECK if c != hero[0] and c != hero[1]]
            drawn = rng.sample(deck, 7)
            villain, board = drawn[:2], drawn[2:]
            h = evaluate(list(hero) + board)
            v = evaluate(villain + board)
            total += 1.0 if h > v else (0.5 if h == v else 0.0)
        table[cls] = total / samples_per_class
        print(f'{i + 1:3d}/169 {cls:4s} {table[cls]:.4f}', flush=True)
    return table


_PREFLOP_EQUITY = None


def preflop_equity_table():
    global _PREFLOP_EQUITY
    if _PREFLOP_EQUITY is None:
        if not os.path.isfile(PREFLOP_TABLE_PATH):
            raise FileNotFoundError(
                f'{PREFLOP_TABLE_PATH} is missing; build it with '
                f'`python equity_agent.py --build-preflop`')
        with open(PREFLOP_TABLE_PATH) as f:
            _PREFLOP_EQUITY = json.load(f)['equity_vs_random']
    return _PREFLOP_EQUITY


class OpponentRange(object):
    ''' The top `fraction` of the 1326 hole-card combos, ranked by pre-flop
    equity against a random hand. Sampling skips combos that collide with
    known cards. '''

    def __init__(self, fraction):
        table = preflop_equity_table()
        ranked = sorted(ALL_COMBOS, key=lambda c: table[hand_class(*c)], reverse=True)
        keep = max(1, int(round(fraction * len(ranked))))
        self.fraction = fraction
        self.combos = ranked[:keep]

    def sample(self, rng, dead):
        while True:
            a, b = self.combos[int(rng.random() * len(self.combos))]
            if a not in dead and b not in dead:
                return a, b

    def live_combos(self, dead):
        return [(a, b) for a, b in self.combos if a not in dead and b not in dead]


# ---------------------------------------------------------------------------
# Equity
# ---------------------------------------------------------------------------

def equity_monte_carlo(hero, board, opp_range, rng, samples):
    ''' Showdown equity of `hero` (2 ints) given `board` (0-5 ints) against a
    hand drawn from `opp_range`, completing the board at random. Ties count
    half. '''
    dead = set(hero) | set(board)
    missing = 5 - len(board)
    total = 0.0
    for _ in range(samples):
        a, b = opp_range.sample(rng, dead)
        rest = [c for c in FULL_DECK if c not in dead and c != a and c != b]
        full_board = board + rng.sample(rest, missing) if missing else board
        h = evaluate(hero + full_board)
        v = evaluate([a, b] + full_board)
        total += 1.0 if h > v else (0.5 if h == v else 0.0)
    return total / samples


def equity_river_exact(hero, board, opp_range):
    ''' Exact showdown equity on a complete board against every live combo of
    the opponent's range (uniformly weighted). '''
    dead = set(hero) | set(board)
    h = evaluate(hero + board)
    combos = opp_range.live_combos(dead)
    total = 0.0
    for a, b in combos:
        v = evaluate([a, b] + board)
        total += 1.0 if h > v else (0.5 if h == v else 0.0)
    return total / len(combos)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

CALL, RAISE, FOLD, CHECK = 0, 1, 2, 3          # LimitholdemEnv.actions order
ACTION_IDS = {'call': CALL, 'raise': RAISE, 'fold': FOLD, 'check': CHECK}

STREETS = ['preflop', 'flop', 'turn', 'river']
_STREET_OF_BOARD = {0: 0, 3: 1, 4: 2, 5: 3}

DEFAULT_PARAMS = {
    # Equity needed to put in the first raise of a round, per street. Equity
    # is measured against the opponent's (aggression-narrowed) range.
    'raise_threshold': [0.40, 0.60, 0.62, 0.64],
    # Extra equity needed for every raise already made this round (3-bet, ...),
    # per street.
    'reraise_step': [0.12, 0.06, 0.06, 0.06],
    # Equity-realisation factor per street when deciding to call: equity
    # must exceed pot_odds / realisation. Below 1 on early streets because a
    # weak hand will often face further bets before showdown.
    'realisation': [0.80, 0.85, 0.92, 1.0],
    # Probability of betting a weak hand (equity < bluff_max_equity) when
    # checked to / first to act, per street.
    'bluff_frequency': [0.0, 0.25, 0.20, 0.15],
    'bluff_max_equity': 0.35,
    # Opponent range (fraction of starting hands kept) after it has made
    # 0, 1, 2, 3+ raises in the hand.
    'range_fractions': [1.0, 0.80, 0.55, 0.40],
    # Width of the logistic soft thresholds (in equity units).
    'temperature': 0.03,
    # Monte Carlo roll-outs per flop / turn equity estimate (standard error
    # about 0.5 / sqrt(samples)); the noise adds a little extra mixing.
    'samples': 300,
    # Roll-outs per pre-flop equity against a narrowed range (cached per
    # starting-hand class, so computed at most 169 x 3 times per agent).
    'preflop_samples': 4000,
}


# Shared by all EquityAgent instances (equity is deterministic, see
# EquityAgent.equity).
_EQUITY_CACHE = {}
_RANGES = {}
_CLASS_REPRESENTATIVE = {}
for _a, _b in ALL_COMBOS:
    _CLASS_REPRESENTATIVE.setdefault(hand_class(_a, _b), [_a, _b])


def _opponent_range(fraction):
    if fraction not in _RANGES:
        _RANGES[fraction] = OpponentRange(fraction)
    return _RANGES[fraction]


def _sigmoid(x):
    if x < -50:
        return 0.0
    if x > 50:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


class EquityAgent(object):
    ''' Rule-based limit hold'em agent driven by hand equity and pot odds.

    Args:
        num_actions (int): kept for RLCard's agent interface.
        seed (int | None): seed of the agent's private RNG.
        **params: overrides for DEFAULT_PARAMS (see the module docstring).
    '''

    def __init__(self, num_actions=4, seed=None, **params):
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError(f'unknown EquityAgent parameters: {sorted(unknown)}')
        self.use_raw = False
        self.num_actions = num_actions
        self.params = {**DEFAULT_PARAMS, **params}
        # Only the action mixing uses this RNG; equity estimates are seeded
        # from the cards themselves (see equity()).
        self.rng = random.Random(seed)
        preflop_equity_table()   # fail early if the table is missing

    def seed(self, seed):
        ''' Reseed the mixing RNG (used by eval_panel for common random numbers). '''
        self.rng.seed(seed)

    # -- range / equity ---------------------------------------------------

    def equity(self, hole, board, opp_raises=0):
        ''' Equity of `hole` (2 ints) on `board` (list of ints) against an
        opponent who has raised `opp_raises` times this hand.

        A deterministic function of its arguments and the sample counts: the
        Monte Carlo roll-outs are seeded from the cards, so caching (shared by
        all EquityAgent instances) never changes results, and the agent's own
        RNG stream does not depend on what happens to be cached. '''
        fractions = self.params['range_fractions']
        fraction = fractions[min(opp_raises, len(fractions) - 1)]
        if not board and fraction >= 1.0:
            return preflop_equity_table()[hand_class(*hole)]
        if not board:
            # Pre-flop equity against a narrowed range depends (up to tiny
            # card-removal effects) only on the starting-hand class, so it is
            # computed once per class from a fixed representative combo and
            # with many more roll-outs.
            cls = hand_class(*hole)
            key = ('preflop', cls, fraction, self.params['preflop_samples'])
            hole = _CLASS_REPRESENTATIVE[cls]
            samples = self.params['preflop_samples']
        else:
            key = (tuple(sorted(hole)), tuple(sorted(board)), fraction, self.params['samples'])
            samples = self.params['samples']
        cached = _EQUITY_CACHE.get(key)
        if cached is not None:
            return cached
        opp_range = _opponent_range(fraction)
        if len(board) == 5:
            value = equity_river_exact(list(hole), list(board), opp_range)
        else:
            rng = random.Random(zlib.crc32(repr(key).encode()))
            value = equity_monte_carlo(list(hole), list(board), opp_range, rng, samples)
        if len(_EQUITY_CACHE) > 500_000:
            _EQUITY_CACHE.clear()
        _EQUITY_CACHE[key] = value
        return value

    # -- decision -----------------------------------------------------------

    @staticmethod
    def _opponent_raises(action_record):
        ''' Raises the opponent has made this hand. Heads-up limit hold'em in
        RLCard strictly alternates players (also across streets), so the
        player who acted last is the opponent. '''
        if not action_record:
            return 0
        opponent = action_record[-1][0]
        return sum(1 for pid, a in action_record if pid == opponent and a == 'raise')

    def decide(self, raw_obs, action_record):
        ''' Returns (action id, info dict). '''
        p = self.params
        legal = {ACTION_IDS[a] for a in raw_obs['legal_actions']}
        hole = [card_from_str(c) for c in raw_obs['hand']]
        board = [card_from_str(c) for c in raw_obs['public_cards']]
        street = _STREET_OF_BOARD[len(board)]

        all_chips = raw_obs['all_chips']
        pot = sum(all_chips)
        to_call = max(all_chips) - raw_obs['my_chips']
        raises_this_round = raw_obs['raise_nums'][street]
        opp_raises = self._opponent_raises(action_record)

        eq = self.equity(hole, board, opp_raises)
        info = {'equity': eq, 'street': STREETS[street], 'opp_raises': opp_raises}

        temp = p['temperature']
        if RAISE in legal:
            threshold = p['raise_threshold'][street] + p['reraise_step'][street] * raises_this_round
            if self.rng.random() < _sigmoid((eq - threshold) / temp):
                return RAISE, info

        if CHECK in legal:
            # Checked to us (or first to act): never fold for free. Bluff the
            # weakest hands sometimes so a check does not always mean "weak".
            if (RAISE in legal and eq < p['bluff_max_equity']
                    and self.rng.random() < p['bluff_frequency'][street]):
                info['bluff'] = True
                return RAISE, info
            return CHECK, info

        # Facing a bet: continue if equity beats the (realisation-adjusted)
        # pot odds.
        pot_odds = to_call / (pot + to_call)
        required = pot_odds / p['realisation'][street]
        info['required_equity'] = required
        if CALL in legal and self.rng.random() < _sigmoid((eq - required) / temp):
            return CALL, info
        return FOLD, info

    # -- RLCard agent interface -------------------------------------------

    def step(self, state):
        action, _ = self.decide(state['raw_obs'], state.get('action_record', []))
        return action

    def eval_step(self, state):
        return self.decide(state['raw_obs'], state.get('action_record', []))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Equity agent utilities')
    parser.add_argument('--build-preflop', action='store_true',
                        help='(Re)compute preflop_equity.json')
    parser.add_argument('--samples', type=int, default=60_000,
                        help='Monte Carlo samples per starting-hand class')
    args = parser.parse_args()
    if args.build_preflop:
        table = build_preflop_table(args.samples)
        with open(PREFLOP_TABLE_PATH, 'w') as f:
            json.dump({'description': 'Showdown equity of each starting-hand class '
                                      'against a uniformly random hand (ties count '
                                      'half), heads-up, Monte Carlo.',
                       'samples_per_class': args.samples,
                       'equity_vs_random': table}, f, indent=1, sort_keys=True)
        print(f'Wrote {PREFLOP_TABLE_PATH}')
