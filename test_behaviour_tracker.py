''' Tests for behaviour_tracker.py.   Run:  python test_behaviour_tracker.py [-k name] '''

import os
import sys
import time
import random
import argparse
import tempfile
import traceback

import behaviour_tracker as bt
from agent_loading import load_policy
from equity_agent import equity_river_exact, OpponentRange, card_from_str, card_to_str


def ints(text):
    return [card_from_str(t[1].upper() + t[0].upper()) for t in text.split()]


def strs(text):
    return [card_to_str(c) for c in ints(text)]


_CACHE = {}


def match(agent, opponent, deals):
    key = (agent, opponent, deals)
    if key not in _CACHE:
        _CACHE[key] = bt.play_match(load_policy(agent), load_policy(opponent), deals, names=(agent, opponent))
    return _CACHE[key]


def stat(stats, name):
    return stats[name][0]


def test_strength_measures():
    hole, board = strs('Ah Kh'), strs('2c 7d 9h Js Kd')
    full = OpponentRange(1.0)
    assert abs(bt.hand_strength_now(hole, board) - equity_river_exact(ints('Ah Kh'), ints('2c 7d 9h Js Kd'), full)) < 1e-12
    assert abs(bt.hand_equity(hole, board) - bt.hand_strength_now(hole, board)) < 1e-12   # river: equal
    # flop: strength now vs equity with draws (nut flush draw: weak now, decent equity)
    draw, flop = strs('Ah Th'), strs('2h 7h Kc')
    assert bt.hand_strength_now(draw, flop) < bt.hand_equity(draw, flop)
    # exact equity against a known hand is symmetric and sums to one
    a, b = strs('As Ad'), strs('Kc Kd')
    for board in ([], strs('2c 7h 9s'), strs('2c 7h 9s Jd'), strs('2c 7h 9s Jd Kh')):
        ab, ba = bt.equity_vs_hand(a, b, board), bt.equity_vs_hand(b, a, board)
        assert abs(ab + ba - 1) < (0.03 if not board else 1e-12), (board, ab, ba)
    assert bt.equity_vs_hand(a, b, strs('2c 7h 9s Jd Kh')) == 0.0      # set of kings wins
    assert bt.best_hand_name(a, strs('2c 7h 9s Jd Kh')) == 'pair'


def test_recording_bookkeeping():
    hands = match('tag', 'lag', 60)
    assert len(hands) == 120
    for h in hands:
        assert abs(sum(h['payoff_bb'].values())) < 1e-9
        assert sorted(h['positions'].values()) == ['BB', 'SB']
        first = h['decisions'][0]
        assert first['street'] == 'preflop' and first['position'] == 'SB'   # SB acts first pre-flop
        assert h['showdown'] == (h['folded'] is None)
        for d in h['decisions']:
            assert d['action'] in d['legal']
            assert abs(sum(d['probs'].values()) - 1) < 1e-5   # stored rounded to 6 decimals
            assert 0 <= d['equity'] <= 1 and 0 <= d['hs'] <= 1 and 0 <= d['eq_actual'] <= 1
        # duplicate format: both seat orders of every deal are present
    assert sorted((h['deal'], h['swap']) for h in hands) == [(i, s) for i in range(60) for s in (0, 1)]
    a, o = bt.compute_stats(hands, 'agent'), bt.compute_stats(hands, 'opponent')
    assert abs(stat(a, 'win rate (mbb/h)') + stat(o, 'win rate (mbb/h)')) < 1e-9


def test_fixed_bot_statistics():
    maniac = bt.compute_stats(match('maniac', 'station', 80), 'agent')
    station = bt.compute_stats(match('maniac', 'station', 80), 'opponent')
    assert stat(maniac, 'PFR') == 1.0 and stat(maniac, 'VPIP') == 1.0
    assert stat(maniac, 'SB open raise') == 1.0 and stat(maniac, 'limp (SB completes)') == 0.0
    for street in ('flop', 'turn', 'river'):
        assert stat(station, f'{street}: fold to bet') == 0.0
        assert stat(station, f'{street}: raise when bet into') == 0.0
    assert stat(station, 'PFR') == 0.0
    assert stat(station, 'WTSD') == 1.0            # nobody ever folds
    assert stat(maniac, 'bluff rate (bet/raise | weak hand)') == 1.0
    assert stat(station, 'bluff share of aggression') is None   # never bets


def test_expected_matches_sampled():
    ''' Probability-weighted frequencies agree with the sampled actions. '''
    hands = match('tag', 'lag', 150)
    first_sb = [d for h in hands for d in h['decisions']
                if d['role'] == 'agent' and d['street'] == 'preflop' and d['position'] == 'SB' and d['raises_round'] == 0]
    expected = sum(d['probs']['raise'] for d in first_sb) / len(first_sb)
    sampled = sum(d['action'] == 'raise' for d in first_sb) / len(first_sb)
    se = (expected * (1 - expected) / len(first_sb)) ** 0.5
    assert abs(expected - sampled) < 4 * se + 0.01, (expected, sampled)
    # and the TAG's open-raise rate matches its calibration (~80%)
    assert 0.7 < expected < 0.9, expected


def test_styles_ordered():
    ''' The LAG is more aggressive and bluffs more than the TAG. '''
    hands = match('tag', 'lag', 150)
    tag, lag = bt.compute_stats(hands, 'agent'), bt.compute_stats(hands, 'opponent')
    assert stat(lag, 'PFR') > stat(tag, 'PFR')
    assert stat(lag, 'bluff rate (bet/raise | weak hand)') > stat(tag, 'bluff rate (bet/raise | weak hand)')
    assert stat(lag, 'flop: bet when checked to') > stat(tag, 'flop: bet when checked to')


def test_save_load_filters_render():
    hands = match('tag', 'lag', 60)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'hands.jsonl')
        bt.save_hands(hands, path)
        assert bt.load_hands(path) == hands
    for name in bt.FILTERS:
        for h in bt.select_hands(hands, name, 'agent', 10):
            assert bt.FILTERS[name](h, 'agent')
            bt.render_hand(h)
    big = bt.select_hands(hands, 'big_pot', 'agent', 5)
    assert [h['pot_bb'] for h in big] == sorted([h['pot_bb'] for h in big], reverse=True)
    html = bt.render_html(hands[:5])
    assert html.startswith('<!doctype html>') and html.count('<pre>') == 5
    bt.format_stats(bt.compute_stats(hands))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', default='')
    args = parser.parse_args()
    tests = [(n, f) for n, f in list(globals().items())
             if n.startswith('test_') and callable(f) and args.k in n]
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
