import numpy as np
from collections import Counter
from itertools import combinations


# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------

RANK_MAP = {
    'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
    '2': 2,  '3': 3,  '4': 4,  '5': 5,  '6': 6,
    '7': 7,  '8': 8,  '9': 9,
}

def parse_card(card_str):
    """Parse an RLCard card string like 'SA', 'HK', 'D2', 'CT' into (suit, rank_int)."""
    suit = card_str[0]   # S H D C
    rank = card_str[1:]  # A K Q J T 2-9
    return suit, RANK_MAP[rank]


# ---------------------------------------------------------------------------
# Hand evaluator  (standard 5-card rank, higher = better)
# Returns a tuple that sorts correctly with Python's default tuple comparison.
# ---------------------------------------------------------------------------

def _hand_rank(cards):
    """
    Evaluate a list of exactly 5 (suit, rank) tuples.
    Returns a comparable tuple (category, tiebreakers...) — higher is better.

    Categories (9=best):
      9 straight flush, 8 four-of-a-kind, 7 full house, 6 flush,
      5 straight, 4 three-of-a-kind, 3 two-pair, 2 one-pair, 1 high card
    """
    suits  = [s for s, _ in cards]
    ranks  = sorted([r for _, r in cards], reverse=True)
    counts = Counter(ranks)
    freq   = sorted(counts.values(), reverse=True)          # e.g. [2,2,1] for two-pair
    groups = sorted(counts.keys(), key=lambda r: (counts[r], r), reverse=True)

    is_flush    = len(set(suits)) == 1
    is_straight = (len(set(ranks)) == 5) and (ranks[0] - ranks[4] == 4)
    # Special case: A-2-3-4-5 wheel straight
    if set(ranks) == {14, 2, 3, 4, 5}:
        is_straight = True
        ranks = [5, 4, 3, 2, 1]   # treat Ace as low for comparison

    if is_straight and is_flush:
        return (9, ranks[0])
    if freq == [4, 1]:
        return (8, groups)
    if freq == [3, 2]:
        return (7, groups)
    if is_flush:
        return (6, ranks)
    if is_straight:
        return (5, ranks[0])
    if freq[0] == 3:
        return (4, groups)
    if freq[:2] == [2, 2]:
        return (3, groups)
    if freq[0] == 2:
        return (2, groups)
    return (1, ranks)


def best_hand_rank(hole_cards, community_cards):
    """
    Given hole cards and community cards (each a list of card strings),
    return the best 5-card rank tuple from all available combinations.
    Returns (0,) if there are fewer than 5 cards (pre-flop with only 2 hole cards).
    """
    all_cards = [parse_card(c) for c in hole_cards + community_cards]
    if len(all_cards) < 5:
        # Pre-flop / early street: fall back to a simple high-card heuristic
        ranks = sorted([r for _, r in all_cards], reverse=True)
        return (1, ranks)   # treat as high-card for comparison purposes

    return max(_hand_rank(list(combo)) for combo in combinations(all_cards, 5))


# ---------------------------------------------------------------------------
# Hand strength  →  [0.0, 1.0]
# ---------------------------------------------------------------------------

def hand_strength(raw_obs):
    """
    Estimate hand strength in [0, 1] from raw_obs.

    raw_obs keys (limit Hold'em):
        'hand'           : list of card strings, e.g. ['SA', 'HK']
        'public_cards'   : list of community card strings (may be empty)
    """
    hole       = raw_obs.get('hand', [])
    community  = raw_obs.get('public_cards', [])

    # Normalise: some versions wrap hole cards in a list, others give a bare string
    if isinstance(hole, str):
        hole = [hole]

    rank = best_hand_rank(hole, community)
    category = rank[0]   # 1–9

    # --- Pre-flop (no community cards): use a simple hole-card heuristic ---
    if not community:
        ranks = sorted([RANK_MAP[c[1:]] for c in hole], reverse=True)
        is_pair   = len(ranks) == 2 and ranks[0] == ranks[1]
        suited    = len(hole) == 2 and hole[0][0] == hole[1][0]
        connected = len(ranks) == 2 and abs(ranks[0] - ranks[1]) == 1
 
        # Base score from the high card alone: 2→0.0, Ace→1.0, scaled to [0, 0.35]
        high_bonus = (ranks[0] - 2) / 12.0 * 0.35
 
        # Kicker bonus: rewards having two high cards, not just one
        kicker_bonus = (ranks[-1] - 2) / 12.0 * 0.10 if len(ranks) > 1 else 0.0
 
        pair_bonus      = 0.40 if is_pair else 0.0
        suit_bonus      = 0.05 if suited else 0.0
        connected_bonus = 0.03 if connected else 0.0
 
        # No base offset — hands must earn their score entirely from card quality
        strength = min(1.0, high_bonus + kicker_bonus + pair_bonus + suit_bonus + connected_bonus)
        return round(strength, 4)

    # --- Post-flop: normalise hand category to [0, 1] ---
    # Category 1 (high card) ~ 0.1, category 9 (straight flush) ~ 1.0
    return round((category - 1) / 8.0, 4)


# ---------------------------------------------------------------------------
# Action mapping for limit Hold'em
#
# These ids must match LimitholdemEnv.actions, which is:
#     ['call', 'raise', 'fold', 'check']  ->  call=0, raise=1, fold=2, check=3
#
# NOTE: these constants used to read FOLD=0, CALL=1, RAISE=2, which inverted
# the agent's decisions against RLCard's ordering: it called its trash hands
# (0 = 'call') and folded its monsters (2 = 'fold'). Any result measured
# against this agent before the fix was measured against an opponent playing
# roughly the opposite of its intended strategy.
#
# Note also that 'call' and 'check' are mutually exclusive in limit Hold'em --
# get_legal_actions() removes 'check' when there is a bet to match and removes
# 'call' when there is not, so exactly one of them is legal at any decision
# point. That is why the "check/call" bands below go through _passive().
# ---------------------------------------------------------------------------

CALL  = 0
RAISE = 1
FOLD  = 2
CHECK = 3


def _passive(available):
    """
    The "check/call" action: stay in the hand as cheaply as possible.
    Calls when there is a bet to match, checks when staying in is free, and
    only folds if neither is legal.
    """
    if CALL in available:
        return CALL
    if CHECK in available:
        return CHECK
    return FOLD


def _choose_action(strength, legal_actions):
    """
    Map a [0,1] strength score to an action index.

    Thresholds (tunable):
      < 0.20  → fold  (very weak hand)
      0.20–0.45 → call  (marginal)
      0.45–0.65 → call/raise  (decent — raise only if raise is legal)
      > 0.65  → raise (strong hand)
    """
    available = set(legal_actions.keys())

    if strength < 0.20:
        action = FOLD if FOLD in available else _passive(available)
    elif strength < 0.45:
        action = _passive(available)
    elif strength < 0.65:
        # Raise with the stronger half of this band
        if RAISE in available and strength > 0.55:
            action = RAISE
        else:
            action = _passive(available)
    else:
        if RAISE in available:
            action = RAISE
        else:
            action = _passive(available)

    return action


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class HonestAgent(object):
    """
    Rule-based agent for Limit Hold'em that estimates hand strength from
    raw_obs and maps it to fold / call / raise decisions.

    use_raw = False so RLCard passes encoded state dicts; we read raw_obs
    from inside the state dict ourselves.
    """

    def __init__(self, num_actions):
        self.use_raw    = False
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        """Called during training (is_training=True). Returns an action int."""
        raw      = state['raw_obs']
        strength = hand_strength(raw)
        return _choose_action(strength, state['legal_actions'])

    def eval_step(self, state):
        """Called during evaluation (is_training=False). Returns (action, info)."""
        raw      = state['raw_obs']
        strength = hand_strength(raw)
        action   = _choose_action(strength, state['legal_actions'])
        info     = {'hand_strength': strength}
        return action, info


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    agent = HonestAgent(num_actions=4)

    # Simulate a few states manually
    test_cases = [
        # (hand, community, description)
        (['SA', 'HA'],  [],              'Pre-flop pocket Aces'),
        (['S2', 'H7'],  [],              'Pre-flop 2-7 offsuit (trash)'),
        (['SA', 'HA'],  ['DA', 'CA', 'SK'], 'Four of a kind Aces'),
        (['S2', 'H7'],  ['D3', 'C4', 'S5'], 'Straight 2-6'),
        (['SA', 'HK'],  ['DA', 'CA', 'SK'], 'Full house A over K'),
        (['S2', 'H7'],  ['D9', 'CJ', 'SQ'], 'Nothing / high card'),
    ]

    # Facing a bet: call/raise/fold are legal, check is not
    legal = {CALL: None, RAISE: None, FOLD: None}

    print(f"{'Hand':<20} {'Community':<25} {'Strength':>8}  Action")
    print('-' * 70)
    for hand, comm, desc in test_cases:
        raw  = {'hand': hand, 'public_cards': comm}
        s    = hand_strength(raw)
        act  = _choose_action(s, legal)
        aname = {CALL: 'call', RAISE: 'raise', FOLD: 'fold', CHECK: 'check'}[act]
        print(f"{desc:<45} {s:>8.4f}  {aname}")
