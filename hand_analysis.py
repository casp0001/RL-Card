''' How good was each decision of a recorded hand? The mistake measure is the
EV loss: the expected value of the best action minus the expected value of
the agent's own action mix,

    EV loss = max_a Q(a) - sum_a p(a) Q(a)        (big blinds, >= 0;
              the Monte Carlo estimate is cross-fitted, so it can come out
              slightly negative when the actions are about equally good)

where Q(a) is the value of taking action a and then playing on as both
players actually play. Q is computed from what the agent knows at the
decision (its own cards, the board, the actions so far), not from the
opponent's actual cards.

  limit   against the actual opponent (e.g. tag), by Monte Carlo: the
          opponent's hole cards are weighted exactly by Bayes' rule (every
          possible holding is replayed through the hand and weighted by the
          opponent policy's probability of the actions it took), then
          `samples` holdings are drawn from that posterior with fresh
          future board cards, and every legal action is played out with
          both policies (common random numbers across the actions of a
          sample, so their differences are precise). A deliberately mixing
          agent has some EV loss against a fixed opponent by construction
          (against a fixed opponent the best action is almost always pure):
          this measures how well the agent exploits that opponent.
  leduc   exact, against the CFR+ equilibrium: the opponent's card is
          weighted by the equilibrium's action probabilities, and the game
          tree is solved for every action. The equilibrium's own action
          probabilities at the same information set are stored alongside:
          EV loss here is distance from game-theoretic play.

    python hand_analysis.py limit --hands experiments/final/hands/dqn_vs_tag.jsonl \
        --kind dqn --ckpt <run dir> --opponent tag [--samples 200] --out <file>.analysis.json
    python hand_analysis.py leduc --kind deepcfr --ckpt <run dir> --out <file>.analysis.json [--hands 10]

The output (JSON) is read by hand_viewer.py.
'''

import os
import sys
import json
import time
import random
import argparse
import itertools

import numpy as np

ACTIONS = ['call', 'raise', 'fold', 'check']
ACTION_ID = {a: i for i, a in enumerate(ACTIONS)}
FULL_DECK = [s + r for s in 'SHDC' for r in 'A23456789TJQK']


# --------------------------------------------------------------------------- #
# Limit hold'em
# --------------------------------------------------------------------------- #

class LimitAnalyzer(object):
    ''' Rebuilds a recorded hand in RLCard with chosen opponent cards and
    future board cards, and plays actions out from any decision. '''

    def __init__(self, agent, opponent, samples=300, seed=0):
        import rlcard_patches  # noqa: F401
        import rlcard
        from rlcard.games.base import Card
        self.Card = Card
        self.env = rlcard.make('limit-holdem', config={'seed': seed, 'allow_step_back': True})
        self.policies = {'agent': agent, 'opponent': opponent}
        self.samples = samples
        self.rng = np.random.RandomState(seed)

    def _card(self, index):
        return self.Card(index[0], index[1])

    def _setup(self, hand, opp_hole, board):
        ''' Reset the env with the recorded small blind, the agent's cards,
        opp_hole for the opponent and `board` (5 cards) as the board. '''
        env = self.env
        sb_seat = next(d['seat'] for d in hand['decisions'] if d['position'] == 'SB')
        while True:
            env.reset()
            if env.game.players[sb_seat].in_chips == 1:
                break
        mover = hand['decisions'][0]
        seats = {mover['role']: mover['seat'],
                 ('opponent' if mover['role'] == 'agent' else 'agent'): 1 - mover['seat']}
        for role, seat in seats.items():
            cards = hand['holes']['agent'] if role == 'agent' else opp_hole
            env.game.players[seat].hand = [self._card(c) for c in cards]
        used = set(hand['holes']['agent']) | set(opp_hole) | set(board)
        rest = [self._card(c) for c in FULL_DECK if c not in used]
        env.game.dealer.deck = rest + [self._card(c) for c in reversed(board)]   # pop() deals from the end
        return seats

    def _probs(self, role, seat):
        state = self.env.get_state(seat)
        p = np.asarray(self.policies[role].probs(state), dtype=float)
        mask = np.zeros(4)
        mask[list(state['legal_actions'])] = 1
        p = p * mask
        return p / p.sum() if p.sum() > 0 else mask / mask.sum()

    def posterior(self, hand):
        ''' For every decision index k: {opponent holding: weight} from the
        opponent's actions before k (holdings that clash with the cards the
        agent has seen by then are excluded). '''
        decisions = hand['decisions']
        known_board_at = [d['board'] for d in decisions]
        mine = set(hand['holes']['agent'])
        # a hand that ended early never dealt the whole board: pad it (the replay never deals the pad)
        full_board = hand['board'] + [c for c in FULL_DECK
                                      if c not in hand['board'] and c not in mine][:5 - len(hand['board'])]
        weights = [dict() for _ in decisions]
        for combo in itertools.combinations([c for c in FULL_DECK if c not in mine], 2):
            # replay as far as the combo is consistent with the board dealt so far
            first_clash = next((k for k, b in enumerate(known_board_at) if set(combo) & set(b)), len(decisions))
            if first_clash == 0:
                continue
            board = [c for c in full_board]
            if set(combo) & set(board):
                # the combo holds a card dealt later: the replay never gets there, any
                # non-clashing filler keeps the deck valid up to the clash
                filler = [c for c in FULL_DECK if c not in mine | set(combo) | set(board)]
                board = [c if c not in combo else filler.pop() for c in board]
            seats = self._setup(hand, list(combo), board)
            w = 1.0
            for k, d in enumerate(decisions[:first_clash]):
                weights[k][combo] = w
                if d['role'] == 'opponent':
                    w *= self._probs('opponent', d['seat'])[ACTION_ID[d['action']]]
                    if w == 0:
                        break
                self.env.step(ACTION_ID[d['action']])
        return weights

    def analyze(self, hand):
        post = self.posterior(hand)
        out = []
        for k, d in enumerate(hand['decisions']):
            if d['role'] != 'agent' or len(d['legal']) < 2:
                out.append(None)
                continue
            combos = [c for c, w in post[k].items() if w > 0]
            w = np.array([post[k][c] for c in combos])
            if not combos:
                out.append(None)
                continue
            draws = self.rng.choice(len(combos), size=self.samples, p=w / w.sum())
            legal = d['legal']
            values = np.zeros((self.samples, len(legal)))
            for i, j in enumerate(draws):
                opp = list(combos[j])
                known = d['board']
                rest = [c for c in FULL_DECK if c not in set(hand['holes']['agent']) | set(opp) | set(known)]
                future = list(self.rng.choice(rest, size=5 - len(known), replace=False))
                seats = self._setup(hand, opp, list(known) + future)
                for prev in hand['decisions'][:k]:
                    self.env.step(ACTION_ID[prev['action']])
                agent_seat = d['seat']
                for ai, a in enumerate(legal):
                    rollout_seed = int(self.rng.randint(2 ** 31)) if ai == 0 else rollout_seed
                    r = np.random.RandomState(rollout_seed)
                    depth = 1
                    self.env.step(ACTION_ID[a])
                    while not self.env.is_over():
                        seat = self.env.get_player_id()
                        role = 'agent' if seat == agent_seat else 'opponent'
                        p = self._probs(role, seat)
                        self.env.step(int(r.choice(4, p=p)))
                        depth += 1
                    values[i, ai] = self.env.get_payoffs()[agent_seat]
                    for _ in range(depth):
                        self.env.step_back()
            out.append(_summary(legal, d['probs'], values, ess=_ess(w), candidates=len(combos)))
        return out


def _ess(w):
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if w.sum() > 0 else 0.0


def _summary(legal, probs, values, **extra):
    ''' Q values, EV loss and their standard errors from per-sample values
    (samples x legal actions, common random numbers). '''
    p = np.array([probs[a] for a in legal]) if probs else None
    q = values.mean(0)
    best = int(q.argmax())
    n = len(values)
    se = values.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(len(legal))
    out = {'values': {a: float(v) for a, v in zip(legal, q)}, 'values_se': {a: float(s) for a, s in zip(legal, se)},
           'best': legal[best], 'samples': n}
    if p is not None:
        p = p / p.sum()
        out['ev'] = float(q @ p)
        if n >= 20:
            # cross-fitting: the max of noisy estimates is biased upwards, so each half of the
            # samples picks the best action and the other half measures the loss against it
            h = n // 2
            first, second = values[:h], values[h:]
            diff = np.concatenate([second[:, first.mean(0).argmax()] - second @ p,
                                   first[:, second.mean(0).argmax()] - first @ p])
        else:
            diff = values[:, best] - values @ p
        out['ev_loss'] = float(diff.mean())
        out['ev_loss_se'] = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
    out.update(extra)
    return out


def cmd_limit(a):
    import torch
    from agent_loading import load_policy
    torch.set_num_threads(a.threads)
    with open(a.hands, encoding='utf-8') as f:
        hands = [json.loads(line) for line in f if line.strip()]
    if a.select:
        # only the selected hands, in the selection's order, with its category labels
        with open(a.select, encoding='utf-8') as f:
            selection = json.load(f)
        by_key = {(h['deal'], h['swap']): h for h in hands}
        hands = [dict(by_key[(s['deal'], s['swap'])], category=s.get('category')) for s in selection]
    analyzer = LimitAnalyzer(load_policy(a.kind, a.ckpt), load_policy(a.opponent), samples=a.samples, seed=a.seed)
    result = {'game': 'limit', 'agent': {'kind': a.kind, 'ckpt': a.ckpt}, 'opponent': a.opponent,
              'measure': 'EV loss against the actual opponent (Monte Carlo, Bayesian opponent range)',
              'samples': a.samples, 'hands': []}
    start = time.perf_counter()
    for i, h in enumerate(hands, 1):
        analysis = analyzer.analyze(h)
        for d, an in zip(h['decisions'], analysis):
            d['analysis'] = an
        result['hands'].append(h)
        print(f'  hand {i}/{len(hands)} analysed ({time.perf_counter() - start:.0f}s)', flush=True)
    _write(a.out, result)


# --------------------------------------------------------------------------- #
# Leduc
# --------------------------------------------------------------------------- #

def _leduc_values(L, history, seat, my_rank, public, mine, opp, with_reach=False):
    ''' Exact Q(a) of the player in `seat` at the state after `history`:
    the opponent's rank (and the public card if not yet dealt) is weighted
    by the prior and by the opponent policy's probabilities of its actions
    so far; play continues with `mine` / `opp`. with_reach: also return the
    probability (over the cards it does not know) that the opponent plays
    into this state. '''
    state = L.LeducState()
    for a in history:
        state = state.child(a)
    totals = {a: 0.0 for a in state.legal_actions()}
    norm = 0.0
    for r_opp in L.RANKS:
        w_opp = L._rank_weights((my_rank,))[r_opp]
        if w_opp == 0:
            continue
        publics = [public] if public is not None else list(L.RANKS)
        for pub in publics:
            # P(public | both private cards): also when the public card is known, since it is
            # evidence about the opponent's card (Leduc has two cards of each rank)
            w_pub = L._rank_weights((my_rank, r_opp))[pub]
            if w_pub == 0:
                continue
            # the opponent's reach: replay the history with its policy
            w, s = w_opp * w_pub, L.LeducState()
            for a in history:
                if s.pointer != seat:
                    p = opp(s, r_opp, pub if s.round == 1 else None)
                    w *= p[a]
                s = s.child(a)
            if w == 0:
                continue
            ranks = (my_rank, r_opp) if seat == 0 else (r_opp, my_rank)
            tables = (mine, opp) if seat == 0 else (opp, mine)
            for a in totals:
                v = L._match_value(state.child(a), ranks, pub, tables)
                totals[a] += w * (v if seat == 0 else -v)
            norm += w
    values = {a: v / norm for a, v in totals.items()}
    return (values, norm) if with_reach else values


def cmd_leduc(a):
    import torch
    import leduc as L
    from agent_loading import load_policy
    torch.set_num_threads(a.threads)
    agent = L._Table(L.policy_of(load_policy(a.kind, a.ckpt)))
    eq = L._Table(L.equilibrium(), True)
    rng = np.random.RandomState(a.seed)
    names = ['J', 'Q', 'K']
    hands = []
    for deal in range(a.deals):
        ranks = list(rng.choice([0, 0, 1, 1, 2, 2], size=3, replace=False))
        seat = deal % 2                                   # the agent's position: 0 = small blind
        my_rank, opp_rank, public = int(ranks[0]), int(ranks[1]), int(ranks[2])
        state, history, decisions = L.LeducState(), [], []
        while not state.is_terminal():
            mover = state.pointer
            pub = public if state.round == 1 else None
            if mover == seat:
                p = agent(state, my_rank, pub)
                values, reach = _leduc_values(L, history, seat, my_rank, pub, agent, eq, with_reach=True)
                q = np.array([values[x] for x in state.legal_actions()])
                pp = np.array([p[x] for x in state.legal_actions()])
                best = int(q.argmax())
                peq = eq(state, my_rank, pub)
                analysis = {'values': {ACTIONS[x]: float(v) for x, v in values.items()},
                            'values_se': {ACTIONS[x]: 0.0 for x in values}, 'best': ACTIONS[state.legal_actions()[best]],
                            'ev': float(q @ pp), 'ev_loss': float(q[best] - q @ pp), 'ev_loss_se': 0.0,
                            'equilibrium_probs': {ACTIONS[x]: float(peq[x]) for x in state.legal_actions()},
                            # how likely the equilibrium opponent plays into this state; where it almost
                            # never does, even the equilibrium's own choice can have a large EV loss
                            # (Nash play only has to be optimal where it is reached)
                            'opponent_reach': float(reach)}
            else:
                p = eq(state, opp_rank, pub)
                analysis = None
            act = int(rng.choice(L.NUM_ACTIONS, p=p))
            decisions.append({'role': 'agent' if mover == seat else 'opponent', 'seat': mover,
                              'position': 'SB' if mover == 0 else 'BB',
                              'street': 'round 1' if state.round == 0 else 'round 2',
                              'board': [names[public]] if state.round == 1 else [],
                              'legal': [ACTIONS[x] for x in state.legal_actions()],
                              'action': ACTIONS[act], 'probs': {ACTIONS[x]: float(p[x]) for x in range(4)},
                              'analysis': analysis})
            history.append(act)
            state = state.child(act)
        payoff = state.payoff(seat, (my_rank, opp_rank) if seat == 0 else (opp_rank, my_rank), public)
        folded = next((d['role'] for d in decisions if d['action'] == 'fold'), None)
        hands.append({'deal': deal, 'names': {'agent': a.name or a.kind, 'opponent': 'CFR+'},
                      'positions': {'agent': 'SB' if seat == 0 else 'BB', 'opponent': 'BB' if seat == 0 else 'SB'},
                      'holes': {'agent': [names[my_rank]], 'opponent': [names[opp_rank]]},
                      'board': [names[public]] if any(d['street'] == 'round 2' for d in decisions) else [],
                      'decisions': decisions, 'showdown': folded is None, 'folded': folded,
                      'payoff_bb': {'agent': float(payoff), 'opponent': -float(payoff)}})
    # the hands worth showing: the largest single-decision EV losses, then some typical ones
    def worst(h):
        return max((d['analysis']['ev_loss'] for d in h['decisions'] if d['analysis']), default=0)
    ranked = sorted(hands, key=worst, reverse=True)
    chosen = [dict(h, category='largest mistake') for h in ranked[:a.hands // 2]]
    rest = [h for h in hands if h not in ranked[:a.hands // 2]]
    chosen += [dict(rest[i], category='random hand')
               for i in sorted(rng.choice(len(rest), size=a.hands - len(chosen), replace=False))]
    losses = [d['analysis']['ev_loss'] for h in hands for d in h['decisions'] if d['analysis']]
    result = {'game': 'leduc', 'agent': {'kind': a.kind, 'ckpt': a.ckpt}, 'opponent': 'CFR+ equilibrium',
              'measure': 'exact EV loss against the CFR+ equilibrium',
              'summary': {'deals': a.deals, 'decisions': len(losses), 'mean_ev_loss': float(np.mean(losses)),
                          'share_with_loss_over_0.05bb': float(np.mean(np.array(losses) > 0.05))},
              'hands': chosen}
    _write(a.out, result)


def _write(path, result):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1)
    print(f'{path}: {len(result["hands"])} hands', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    lim = sub.add_parser('limit')
    lim.add_argument('--hands', required=True, help='hands .jsonl from behaviour_tracker.play_match')
    lim.add_argument('--select', default=None, help='optional JSON list of hands: analyse only these')
    lim.add_argument('--kind', required=True)
    lim.add_argument('--ckpt', default=None)
    lim.add_argument('--opponent', default='tag')
    lim.add_argument('--samples', type=int, default=300, help='opponent holdings drawn per decision')
    led = sub.add_parser('leduc')
    led.add_argument('--kind', required=True)
    led.add_argument('--ckpt', required=True)
    led.add_argument('--name', default=None)
    led.add_argument('--deals', type=int, default=400, help='hands played against the equilibrium')
    led.add_argument('--hands', type=int, default=10, help='hands kept: the worst ones and typical ones')
    for q in (lim, led):
        q.add_argument('--out', required=True)
        q.add_argument('--seed', type=int, default=0)
        q.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    {'limit': cmd_limit, 'leduc': cmd_leduc}[a.command](a)


if __name__ == '__main__':
    main()
