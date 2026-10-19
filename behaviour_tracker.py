''' Behaviour tracker: plays an agent against an opponent on duplicate deals,
records every decision with its context (cards, position, pot, the agent's
action probabilities, hand strength), and computes playing-style statistics
-- including how often an agent bluffs -- plus readable hand replays.

Commands:
    python behaviour_tracker.py track --kind nfsp --ckpt experiments/nfsp/<run> --opponent tag \
        --deals 1000 --out experiments/behaviour/nfsp_vs_tag [--show 5 --filter river_bluff] [--html]
    python behaviour_tracker.py replay experiments/behaviour/nfsp_vs_tag/hands.jsonl --filter bluff --show 5
    python behaviour_tracker.py stats  experiments/behaviour/nfsp_vs_tag/hands.jsonl [--role opponent]
    python behaviour_tracker.py compare dqn=.../stats.json nfsp=.../stats.json ...   (side-by-side table)

Kinds (agent and --opponent) are those of agent_loading.py: dqn nfsp deepcfr ppo
(with --ckpt / --opp_ckpt) and tag nit lag honest station maniac random.

Hand-strength measures stored with every decision (all against the hand's
own cards, ties count half):
    equity       showdown equity against a uniformly random hand, averaged
                 over the remaining board cards (includes drawing chances)
    hs           strength right now: share of random hands the current
                 best hand beats on the current board (no more cards);
                 equals equity pre-flop and on the river
    eq_actual    equity against the opponent's ACTUAL hand (exact on the
                 flop / turn / river by enumerating the run-outs)

Style statistics (the probability-weighted ones use the policy's action
probabilities at the decisions that were reached, which is much less noisy
than counting the sampled actions):
    VPIP, PFR                  share of hands with a voluntary call / raise pre-flop
    3-bet                      P(raise) facing one pre-flop raise
    limp                       small blind's P(call) on its first decision
    bet / fold / raise per street   P(bet | checked to), P(fold | facing a bet),
                               P(raise | facing a bet) on flop, turn, river
    aggression factor          post-flop  sum P(bet or raise) / sum P(call)
    c-bet                      pre-flop aggressor's P(bet) on its first flop decision
    WTSD, W$SD                 went to showdown (of hands that saw the flop), won at showdown
    bluff rate                 P(bet or raise | weak hand: equity < 0.35, not a draw)
    bluff share                share of all post-flop aggression made with such weak hands
    semi-bluff share           share of flop / turn aggression with a weak made hand
                               (hs < 0.5) and real drawing equity (equity - hs >= 0.08)
    value share                share of post-flop aggression with equity >= 0.6
    slow-play rate             P(check or call | equity >= 0.8, raising allowed)
    river aggression while behind  share of river bets / raises made while behind the
                               opponent's actual hand (bluffs that got called, or value
                               bets into a better hand)
'''

import os
import sys
import csv
import json
import math
import random
import argparse
from collections import OrderedDict

import numpy as np

import rlcard_patches  # noqa: F401
from equity_agent import (EquityAgent, evaluate, card_from_str, CATEGORY_NAMES, FULL_DECK,
                          ALL_COMBOS)

ACTION_NAMES = ['call', 'raise', 'fold', 'check']
CALL, RAISE, FOLD, CHECK = 0, 1, 2, 3
STREETS = ['preflop', 'flop', 'turn', 'river']
STREET_OF_BOARD = {0: 'preflop', 3: 'flop', 4: 'turn', 5: 'river'}
BIG_BLIND = 2   # chips

THRESHOLDS = {'bluff_equity': 0.35, 'value_equity': 0.6, 'slowplay_equity': 0.8,
              'semi_hs': 0.5, 'semi_potential': 0.08}

_EQUITY = EquityAgent(seed=0)   # only used for its deterministic, cached equity()


# --------------------------------------------------------------------------- #
# Cards and hand strength
# --------------------------------------------------------------------------- #

def card_text(card):
    ''' RLCard 'SA' -> 'As' (rank + lower-case suit). '''
    return card[1] + card[0].lower()


def cards_text(cards):
    return ' '.join(card_text(c) for c in cards) if cards else '--'


def _ints(cards):
    return [card_from_str(c) for c in cards]


def hand_equity(hole, board):
    ''' Equity against a random hand over the remaining board (cached). '''
    return float(_EQUITY.equity(_ints(hole), _ints(board), 0))


def hand_strength_now(hole, board):
    ''' Share of random opponent hands beaten right now (no more cards). '''
    if len(board) < 3:
        return hand_equity(hole, board)
    h, b = _ints(hole), _ints(board)
    dead = set(h) | set(b)
    mine = evaluate(h + b)
    total = n = 0
    for x, y in ALL_COMBOS:
        if x in dead or y in dead:
            continue
        theirs = evaluate([x, y] + b)
        total += 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
        n += 1
    return total / n


def equity_vs_hand(hole, other, board, samples=2000, seed=0):
    ''' Equity of `hole` against the known hand `other`: exact on the flop,
    turn and river (all run-outs), Monte Carlo pre-flop. '''
    h, o, b = _ints(hole), _ints(other), _ints(board)
    rest = [c for c in FULL_DECK if c not in h and c not in o and c not in b]
    missing = 5 - len(b)

    def outcome(full):
        x, y = evaluate(h + full), evaluate(o + full)
        return 1.0 if x > y else (0.5 if x == y else 0.0)
    if missing == 0:
        return outcome(b)
    if missing <= 2:
        from itertools import combinations
        runs = list(combinations(rest, missing))
        return sum(outcome(b + list(r)) for r in runs) / len(runs)
    rng = random.Random(seed)
    return sum(outcome(b + rng.sample(rest, missing)) for _ in range(samples)) / samples


def best_hand_name(hole, board):
    if len(board) < 3:
        return ''
    return CATEGORY_NAMES[evaluate(_ints(hole) + _ints(board)) >> 20]


# --------------------------------------------------------------------------- #
# Recording hands
# --------------------------------------------------------------------------- #

class HandRecorder(object):
    ''' Records one hand at a time from a running RLCard limit hold'em env.

    Call start(env, seat_roles) after env.reset(), decision(...) before every
    env.step(action), and finish(env) once env.is_over(). Roles are labels
    such as 'agent' / 'opponent' or 'you' / 'agent'. '''

    def __init__(self, names):
        self.names = dict(names)      # role -> display name
        self.hand = None

    def start(self, env, seat_roles, deal=None, swap=None):
        g = env.game
        self.seat_roles = dict(seat_roles)
        positions = {}
        holes = {}
        for seat, role in self.seat_roles.items():
            positions[role] = 'SB' if g.players[seat].in_chips == 1 else 'BB'
            holes[role] = [c.get_index() for c in g.players[seat].hand]
        self.hand = {'deal': deal, 'swap': swap, 'names': self.names, 'positions': positions,
                     'holes': holes, 'decisions': [], 'preflop_aggressor': None}

    def decision(self, env, state, seat, action, probs=None):
        role = self.seat_roles[seat]
        other = next(r for s, r in self.seat_roles.items() if s != seat)
        raw = state['raw_obs']
        hole, board = list(raw['hand']), list(raw['public_cards'])
        street = STREET_OF_BOARD[len(board)]
        legal = [ACTION_NAMES[a] for a in state['legal_actions']]
        to_call = max(raw['all_chips']) - raw['my_chips']
        record = {
            'role': role, 'seat': seat, 'position': self.hand['positions'][role], 'street': street,
            'hole': hole, 'board': board,
            'pot_bb': sum(raw['all_chips']) / BIG_BLIND, 'to_call_bb': to_call / BIG_BLIND,
            'raises_round': raw['raise_nums'][STREETS.index(street)],
            'legal': legal, 'action': ACTION_NAMES[int(action)],
            'probs': ({ACTION_NAMES[a]: round(float(probs[a]), 6) for a in range(4)}
                      if probs is not None else None),
            'equity': round(hand_equity(hole, board), 4),
            'hs': round(hand_strength_now(hole, board), 4),
            'eq_actual': round(equity_vs_hand(hole, self.hand['holes'][other], board,
                                              seed=len(self.hand['decisions'])), 4),
        }
        if street == 'preflop' and record['action'] == 'raise':
            self.hand['preflop_aggressor'] = role
        self.hand['decisions'].append(record)
        return record

    def finish(self, env):
        g = env.game
        payoffs = env.get_payoffs()
        folded = [s for s in self.seat_roles if 'fold' in str(g.players[s].status).lower()]
        board = [c.get_index() for c in g.public_cards]
        hand = self.hand
        hand['board'] = board
        hand['showdown'] = not folded
        hand['folded'] = self.seat_roles[folded[0]] if folded else None
        hand['payoff_bb'] = {self.seat_roles[s]: float(payoffs[s]) for s in self.seat_roles}
        hand['pot_bb'] = sum(p.in_chips for p in g.players) / BIG_BLIND
        if hand['showdown']:
            hand['best_hands'] = {r: best_hand_name(hand['holes'][r], board) for r in hand['holes']}
        self.hand = None
        return hand


def play_match(agent, opponent, deals, names=('agent', 'opponent'), seed=30_000_000, progress=False):
    ''' Duplicate deals (each played twice with seats swapped) between two
    Policies (agent_loading.py); returns the recorded hands. '''
    import torch
    import rlcard
    env = rlcard.make('limit-holdem', config={'seed': seed})
    recorder = HandRecorder({'agent': names[0], 'opponent': names[1]})
    policies = {'agent': agent, 'opponent': opponent}
    hands = []
    for i in range(deals):
        for swap in (0, 1):
            s = seed + i
            env.seed(s)
            np.random.seed(s % (2 ** 32))
            random.seed(s)
            torch.manual_seed(s)
            for k, p in enumerate((agent, opponent)):
                if hasattr(p, 'seed'):
                    p.seed(s + 1 + k)
            seat_roles = {0: 'agent', 1: 'opponent'} if swap == 0 else {0: 'opponent', 1: 'agent'}
            state, seat = env.reset()
            recorder.start(env, seat_roles, deal=i, swap=swap)
            while not env.is_over():
                policy = policies[seat_roles[seat]]
                probs = np.asarray(policy.probs(state), dtype=float)
                action, _ = policy.eval_step(state)
                recorder.decision(env, state, seat, action, probs)
                state, seat = env.step(action)
            hands.append(recorder.finish(env))
        if progress and (i + 1) % max(1, deals // 10) == 0:
            print(f'  {i + 1}/{deals} deals', flush=True)
    return hands


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def _p(decision, action):
    ''' Probability (or 0/1 if no probabilities were recorded) of an action. '''
    if decision.get('probs'):
        return decision['probs'][action]
    return 1.0 if decision['action'] == action else 0.0


def _rate(numerator, denominator):
    return (numerator / denominator if denominator else None, denominator)


def classify(d, t=THRESHOLDS):
    ''' Hand class of a post-flop decision: 'semi' / 'weak' / 'value' / 'medium'. '''
    if d['street'] in ('flop', 'turn') and d['hs'] < t['semi_hs'] and d['equity'] - d['hs'] >= t['semi_potential']:
        return 'semi'
    if d['equity'] < t['bluff_equity']:
        return 'weak'
    if d['equity'] >= t['value_equity']:
        return 'value'
    return 'medium'


def compute_stats(hands, role='agent', t=THRESHOLDS):
    ''' Ordered dict  name -> (value, sample size). '''
    stats = OrderedDict()
    decisions = [d for h in hands for d in h['decisions'] if d['role'] == role]

    # win rate (duplicate pairs averaged per deal when available)
    by_deal = {}
    for h in hands:
        by_deal.setdefault(h['deal'], []).append(h['payoff_bb'][role])
    per = np.array([np.mean(v) for v in by_deal.values()]) * 1000
    stats['win rate (mbb/h)'] = (float(per.mean()) if len(per) else None, len(hands))
    stats['win rate SE (mbb/h)'] = (float(per.std(ddof=1) / math.sqrt(len(per))) if len(per) > 1 else None, len(per))

    # pre-flop, hand level (sampled actions)
    vpip = pfr = 0
    for h in hands:
        pre = [d for d in h['decisions'] if d['role'] == role and d['street'] == 'preflop']
        vpip += any(d['action'] in ('call', 'raise') for d in pre)
        pfr += any(d['action'] == 'raise' for d in pre)
    stats['VPIP'] = _rate(vpip, len(hands))
    stats['PFR'] = _rate(pfr, len(hands))
    pre = [d for d in decisions if d['street'] == 'preflop']
    first_sb = [d for d in pre if d['position'] == 'SB' and d['raises_round'] == 0 and 'call' in d['legal']]
    stats['limp (SB completes)'] = _rate(sum(_p(d, 'call') for d in first_sb), len(first_sb))
    stats['SB open raise'] = _rate(sum(_p(d, 'raise') for d in first_sb), len(first_sb))
    three = [d for d in pre if d['raises_round'] == 1 and 'call' in d['legal'] and 'raise' in d['legal']]
    stats['3-bet'] = _rate(sum(_p(d, 'raise') for d in three), len(three))
    facing_pre = [d for d in pre if 'call' in d['legal']]
    stats['preflop fold to raise'] = _rate(sum(_p(d, 'fold') for d in facing_pre if d['raises_round'] >= 1),
                                          sum(1 for d in facing_pre if d['raises_round'] >= 1))

    # post-flop per street
    post = [d for d in decisions if d['street'] != 'preflop']
    for street in ('flop', 'turn', 'river'):
        ds = [d for d in post if d['street'] == street]
        free = [d for d in ds if 'check' in d['legal'] and 'raise' in d['legal']]
        facing = [d for d in ds if 'call' in d['legal']]
        stats[f'{street}: bet when checked to'] = _rate(sum(_p(d, 'raise') for d in free), len(free))
        stats[f'{street}: fold to bet'] = _rate(sum(_p(d, 'fold') for d in facing), len(facing))
        raisable = [d for d in facing if 'raise' in d['legal']]
        stats[f'{street}: raise when bet into'] = _rate(sum(_p(d, 'raise') for d in raisable), len(raisable))
    aggr = sum(_p(d, 'raise') for d in post)
    calls = sum(_p(d, 'call') for d in post)
    stats['aggression factor (post-flop)'] = (aggr / calls if calls else None, len(post))

    # c-bet
    cbet_spots = []
    for h in hands:
        if h['preflop_aggressor'] != role:
            continue
        flop = [d for d in h['decisions'] if d['role'] == role and d['street'] == 'flop']
        if flop and 'check' in flop[0]['legal'] and 'raise' in flop[0]['legal']:
            cbet_spots.append(flop[0])
    stats['c-bet'] = _rate(sum(_p(d, 'raise') for d in cbet_spots), len(cbet_spots))

    # showdowns
    saw_flop = [h for h in hands if any(d['role'] == role and d['street'] != 'preflop' for d in h['decisions'])]
    showdowns = [h for h in saw_flop if h['showdown']]
    stats['WTSD'] = _rate(len(showdowns), len(saw_flop))
    stats['W$SD'] = _rate(sum(1.0 if h['payoff_bb'][role] > 0 else (0.5 if h['payoff_bb'][role] == 0 else 0)
                              for h in showdowns), len(showdowns))

    # bluffing
    can_raise = [d for d in post if 'raise' in d['legal']]
    classes = [classify(d, t) for d in can_raise]
    total_aggr = sum(_p(d, 'raise') for d in can_raise)
    weak = [d for d, c in zip(can_raise, classes) if c == 'weak']
    stats['bluff rate (bet/raise | weak hand)'] = _rate(sum(_p(d, 'raise') for d in weak), len(weak))
    stats['bluff share of aggression'] = _rate(sum(_p(d, 'raise') for d in weak), total_aggr)
    semi = [d for d, c in zip(can_raise, classes) if c == 'semi']
    stats['semi-bluff share of aggression'] = _rate(sum(_p(d, 'raise') for d in semi), total_aggr)
    value = [d for d, c in zip(can_raise, classes) if c == 'value']
    stats['value share of aggression'] = _rate(sum(_p(d, 'raise') for d in value), total_aggr)
    strong = [d for d in can_raise if d['equity'] >= t['slowplay_equity']]
    stats['slow-play rate (no raise | equity >= 0.8)'] = _rate(sum(1 - _p(d, 'raise') for d in strong), len(strong))
    river = [d for d in can_raise if d['street'] == 'river']
    river_aggr = sum(_p(d, 'raise') for d in river)
    stats['river: aggression while behind (share)'] = _rate(
        sum(_p(d, 'raise') for d in river if d['eq_actual'] < 0.5), river_aggr)
    return stats


def format_stats(stats, title=''):
    lines = [title] if title else []
    for name, (value, n) in stats.items():
        if value is None:
            text = 'n/a'
        elif 'mbb' in name:
            text = f'{value:+.1f}'
        elif 'factor' in name:
            text = f'{value:.2f}'
        else:
            text = f'{value:6.1%}'
        size = f'{n:.0f}' if isinstance(n, (int, float)) else str(n)
        lines.append(f'  {name:42s} {text:>9s}   (n={size})')
    return '\n'.join(lines)


def stats_to_json(stats):
    return {k: {'value': v, 'n': n} for k, (v, n) in stats.items()}


# --------------------------------------------------------------------------- #
# Replays
# --------------------------------------------------------------------------- #

def _decision_tags(d, t=THRESHOLDS):
    tags = []
    if d['street'] != 'preflop' and d['action'] == 'raise':
        c = classify(d, t)
        if c in ('weak', 'semi'):
            tags.append('BLUFF' if c == 'weak' else 'SEMI-BLUFF')
        if d['street'] == 'river' and d['eq_actual'] < 0.5:
            tags.append('behind')
    if d['equity'] >= t['slowplay_equity'] and 'raise' in d['legal'] and d['action'] in ('check', 'call'):
        tags.append('SLOW-PLAY')
    return tags


def _action_word(d):
    if d['action'] == 'raise':
        return 'bets' if d['to_call_bb'] == 0 else 'raises'
    if d['action'] == 'call' and d['street'] == 'preflop' and d['raises_round'] == 0:
        return 'limps'
    return {'call': 'calls', 'check': 'checks', 'fold': 'folds'}[d['action']]


def render_hand(hand, show_probs=True):
    names = hand['names']
    roles = list(hand['holes'])
    head = (f"Deal {hand['deal']}" + (f" (seats swapped)" if hand.get('swap') else '') + ': '
            + ', '.join(f"{names[r]} {hand['positions'][r]} [{cards_text(hand['holes'][r])}]" for r in roles))
    lines = [head]
    street = None
    for d in hand['decisions']:
        if d['street'] != street:
            street = d['street']
            board = f"  [{cards_text(d['board'])}]" if d['board'] else ''
            lines.append(f"  {street.upper():8s}{board}   pot {d['pot_bb']:.1f} BB")
        info = f"eq {d['equity']:.2f}"
        if d['street'] != 'preflop':
            info += f" hs {d['hs']:.2f}"
        info += f" vs-hand {d['eq_actual']:.2f}"
        probs = ''
        if show_probs and d.get('probs'):
            probs = '  p(' + ' '.join(f"{a}={d['probs'][a]:.2f}" for a in d['legal']) + ')'
        tags = _decision_tags(d)
        lines.append(f"    {names[d['role']]:>12s} {_action_word(d):7s} ({info}){probs}"
                     + (f"   <{', '.join(tags)}>" if tags else ''))
    if hand['showdown']:
        bh = hand.get('best_hands', {})
        lines.append(f"  SHOWDOWN [{cards_text(hand['board'])}]: "
                     + ', '.join(f"{names[r]} {bh.get(r, '')}" for r in roles))
    else:
        lines.append(f"  {names[hand['folded']]} folded")
    lines.append('  result: ' + ', '.join(f"{names[r]} {hand['payoff_bb'][r]:+.1f} BB" for r in roles))
    return '\n'.join(lines)


FILTERS = {
    'all': lambda h, role: True,
    'showdown': lambda h, role: h['showdown'],
    'bluff': lambda h, role: any(d['role'] == role and 'BLUFF' in _decision_tags(d) for d in h['decisions']),
    'semi_bluff': lambda h, role: any(d['role'] == role and 'SEMI-BLUFF' in _decision_tags(d) for d in h['decisions']),
    'river_bluff': lambda h, role: any(d['role'] == role and d['street'] == 'river' and d['action'] == 'raise'
                                       and classify(d) == 'weak' for d in h['decisions']),
    'river_behind': lambda h, role: any(d['role'] == role and d['street'] == 'river' and d['action'] == 'raise'
                                        and d['eq_actual'] < 0.5 for d in h['decisions']),
    'slowplay': lambda h, role: any(d['role'] == role and 'SLOW-PLAY' in _decision_tags(d) for d in h['decisions']),
    'big_pot': lambda h, role: True,
    'big_win': lambda h, role: h['payoff_bb'][role] > 0,
    'big_loss': lambda h, role: h['payoff_bb'][role] < 0,
}


def select_hands(hands, name='all', role='agent', count=5):
    chosen = [h for h in hands if FILTERS[name](h, role)]
    if name == 'big_pot':
        chosen.sort(key=lambda h: -h['pot_bb'])
    elif name == 'big_win':
        chosen.sort(key=lambda h: -h['payoff_bb'][role])
    elif name == 'big_loss':
        chosen.sort(key=lambda h: h['payoff_bb'][role])
    return chosen[:count]


def render_html(hands, title='Hand replays'):
    import re
    import html as _html
    suits = {'s': ('&#9824;', '#111'), 'h': ('&#9829;', '#c00'), 'd': ('&#9830;', '#c00'), 'c': ('&#9827;', '#111')}

    def colour(text):
        out = _html.escape(text)
        out = re.sub(r'\b([AKQJT2-9])([shdc])\b',
                     lambda m: f'<b style="color:{suits[m.group(2)][1]}">{m.group(1)}{suits[m.group(2)][0]}</b>', out)
        return re.sub(r'&lt;([A-Za-z, -]+)&gt;', r'<span class="tag">&lt;\1&gt;</span>', out)
    blocks = ''.join(f'<pre>{colour(render_hand(h))}</pre>' for h in hands)
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{_html.escape(title)}</title>'
            '<style>body{font-family:Segoe UI,sans-serif;background:#f6f7f9;margin:24px}'
            'pre{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px;font-size:13px;'
            'line-height:1.45;overflow-x:auto}.tag{color:#b45309;font-weight:bold}</style></head>'
            f'<body><h2>{_html.escape(title)}</h2>{blocks}</body></html>')


def load_hands(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_hands(hands, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for h in hands:
            f.write(json.dumps(h) + '\n')


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def _print_replays(hands, a):
    chosen = select_hands(hands, a.filter, a.role, a.show)
    print(f'\n{len(chosen)} hand(s) matching "{a.filter}" (role {a.role}):')
    for h in chosen:
        print('\n' + render_hand(h))
    return chosen


def main():
    from agent_loading import KINDS
    p = argparse.ArgumentParser("Behaviour tracker - limit hold'em")
    sub = p.add_subparsers(dest='command', required=True)

    t = sub.add_parser('track', help='play a match and record every decision')
    t.add_argument('--kind', required=True, choices=KINDS)
    t.add_argument('--ckpt', default=None)
    t.add_argument('--opponent', default='tag', choices=KINDS)
    t.add_argument('--opp_ckpt', default=None)
    t.add_argument('--deals', type=int, default=1000, help='duplicate deals (2 hands each)')
    t.add_argument('--out', default=None, help='output folder (default experiments/behaviour/<kind>_vs_<opponent>)')
    t.add_argument('--show', type=int, default=3, help='hand replays to print')
    t.add_argument('--filter', default='bluff', choices=sorted(FILTERS))
    t.add_argument('--role', default='agent', choices=['agent', 'opponent'])
    t.add_argument('--html', action='store_true', help='also write replays.html')

    r = sub.add_parser('replay', help='print hands from a hands.jsonl file')
    r.add_argument('file')
    r.add_argument('--filter', default='all', choices=sorted(FILTERS))
    r.add_argument('--role', default='agent')
    r.add_argument('--show', type=int, default=5)
    r.add_argument('--html', default=None, help='write the selection to this HTML file')

    s = sub.add_parser('stats', help='statistics of a hands.jsonl file')
    s.add_argument('file')
    s.add_argument('--role', default='agent')

    c = sub.add_parser('compare', help='side-by-side table of several stats.json files (label=path)')
    c.add_argument('entries', nargs='+')
    c.add_argument('--csv', default=None)
    a = p.parse_args()

    if a.command == 'track':
        from agent_loading import load_policy
        agent = load_policy(a.kind, a.ckpt)
        opponent = load_policy(a.opponent, a.opp_ckpt)
        out = a.out or os.path.join('experiments', 'behaviour', f'{a.kind}_vs_{a.opponent}')
        hands = play_match(agent, opponent, a.deals, names=(a.kind, a.opponent), progress=True)
        save_hands(hands, os.path.join(out, 'hands.jsonl'))
        for role in ('agent', 'opponent'):
            stats = compute_stats(hands, role)
            name = a.kind if role == 'agent' else a.opponent
            with open(os.path.join(out, f'stats_{role}.json'), 'w', encoding='utf-8') as f:
                json.dump({'name': name, 'hands': len(hands), 'thresholds': THRESHOLDS,
                           'stats': stats_to_json(stats)}, f, indent=2)
            print('\n' + format_stats(stats, f'{name} ({role}) over {len(hands)} hands:'))
        chosen = _print_replays(hands, a)
        if a.html:
            with open(os.path.join(out, 'replays.html'), 'w', encoding='utf-8') as f:
                f.write(render_html(chosen, f'{a.kind} vs {a.opponent}: {a.filter}'))
        print(f'\nwrote {out}/hands.jsonl, stats_agent.json, stats_opponent.json')
    elif a.command == 'replay':
        hands = load_hands(a.file)
        chosen = _print_replays(hands, a)
        if a.html:
            with open(a.html, 'w', encoding='utf-8') as f:
                f.write(render_html(chosen, f'{os.path.basename(a.file)}: {a.filter}'))
    elif a.command == 'stats':
        hands = load_hands(a.file)
        print(format_stats(compute_stats(hands, a.role), f'{a.role} over {len(hands)} hands:'))
    elif a.command == 'compare':
        columns = []
        for entry in a.entries:
            label, path = entry.split('=', 1)
            with open(path, encoding='utf-8') as f:
                columns.append((label, json.load(f)['stats']))
        names = list(columns[0][1])
        width = max(12, max(len(l) for l, _ in columns) + 2)
        print(f"{'statistic':44s}" + ''.join(f'{l:>{width}s}' for l, _ in columns))
        rows = []
        for n in names:
            cells = []
            for _, st in columns:
                v = st.get(n, {}).get('value')
                cells.append('n/a' if v is None else (f'{v:+.0f}' if 'mbb' in n else
                                                      f'{v:.2f}' if 'factor' in n else f'{v:.1%}'))
            rows.append([n] + cells)
            print(f'{n:44s}' + ''.join(f'{c:>{width}s}' for c in cells))
        if a.csv:
            with open(a.csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['statistic'] + [l for l, _ in columns])
                w.writerows(rows)


if __name__ == '__main__':
    main()
