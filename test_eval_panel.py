''' Tests for eval_panel.py.   Run:  python test_eval_panel.py  [-k name] '''

import os
import sys
import csv
import time
import random
import tempfile
import argparse
import traceback

import numpy as np
import torch

from eval_panel import (Panel, PanelLogger, FixedPolicyAgent, parse_members, eval_mode,
                        preserved_global_rng, DEFAULT_MEMBERS, MEMBERS, format_results)
from equity_agent import EquityAgent, CALL, RAISE, FOLD, CHECK


def test_parse_members():
    assert parse_members('random, tag') == ('random', 'tag')
    try:
        parse_members('random,tga')
    except ValueError:
        return
    raise AssertionError('typo accepted')


def test_common_random_numbers():
    ''' Same agent, same panel -> bit-identical results, also from a fresh
    Panel object, and regardless of the equity cache being warm. '''
    agent = EquityAgent(seed=0)
    r1 = Panel(num_deals=60).evaluate(agent)
    r2 = Panel(num_deals=60).evaluate(agent)
    r3 = Panel(num_deals=60).evaluate(EquityAgent(seed=123))   # agent reseeded per deal
    assert r1['members'] == r2['members'] == r3['members'], (r1['members'], r2['members'])
    assert r1['actions'] == r2['actions']


def test_rng_isolation():
    ''' Evaluation must not change numpy / random / torch global streams. '''
    def draws():
        return (np.random.rand(3).tolist(), [random.random() for _ in range(3)],
                torch.rand(3).tolist())
    np.random.seed(5); random.seed(5); torch.manual_seed(5)
    expected = draws()
    np.random.seed(5); random.seed(5); torch.manual_seed(5)
    Panel(members=('random', 'tag'), num_deals=20).evaluate(EquityAgent(seed=0))
    assert draws() == expected


def test_result_structure():
    panel = Panel(num_deals=30)
    r = panel.evaluate(EquityAgent(seed=0))
    assert list(r['members']) == list(DEFAULT_MEMBERS)
    scored = [v['mean'] for k, v in r['members'].items() if k != 'random']
    assert r['min'] == min(scored) and abs(r['mean'] - np.mean(scored)) < 1e-9
    assert abs(sum(r['actions'].values()) - 1) < 1e-9
    assert set(r['actions']) == {'call', 'raise', 'fold', 'check'}
    format_results(r)   # does not crash


def test_duplicate_cancels_mirror_matches():
    ''' A deterministic policy against itself scores exactly 0 per deal:
    the seat swap replays the identical hand from the other side. '''
    panel = Panel(members=('station', 'maniac'), num_deals=200)
    r_station = panel.evaluate(FixedPolicyAgent([CHECK, CALL]))
    r_maniac = panel.evaluate(FixedPolicyAgent([RAISE, CALL, CHECK]))
    assert r_station['members']['station']['mean'] == 0.0
    assert r_maniac['members']['maniac']['mean'] == 0.0
    # and the two fixed bots are exact negatives of each other
    assert r_station['members']['maniac']['mean'] == -r_maniac['members']['station']['mean']


def test_nfsp_eval_mode():
    class NFSPAgent(object):   # stand-in with the attributes the panel uses
        use_raw = False

        def __init__(self):
            self.evaluate_with = 'best_response'
            self.seen = set()

        def eval_step(self, state):
            self.seen.add(self.evaluate_with)
            legal = state['legal_actions']
            return (CHECK if CHECK in legal else CALL), {}

    agent = NFSPAgent()
    Panel(members=('station',), num_deals=5).evaluate(agent)
    assert agent.seen == {'average_policy'} and agent.evaluate_with == 'best_response'
    agent.seen.clear()
    Panel(members=('station',), num_deals=5).evaluate(agent, nfsp_policy='best_response')
    assert agent.seen == {'best_response'}
    with eval_mode(object()):   # non-NFSP agents are left alone
        pass


def test_panel_logger():
    with tempfile.TemporaryDirectory() as d:
        panel = Panel(members=('random', 'station'), num_deals=10)
        logger = PanelLogger(d, panel.members)
        for ep in (0, 100):
            logger.log(ep, panel.evaluate(FixedPolicyAgent([RAISE, CALL, CHECK])))
        with open(logger.csv_path) as f:
            rows = list(csv.DictReader(f))
        assert [int(r['episode']) for r in rows] == [0, 100]
        assert {'min', 'mean', 'random_mean', 'station_se', 'act_raise'} <= set(rows[0])
        assert os.path.isfile(logger.fig_path)


def test_style_ordering():
    ''' The TAG ("reasonable") member must beat the nit and the lag and break
    even against itself; every Equity style beats the trivial bots. '''
    panel = Panel(num_deals=1500)
    tag = panel.evaluate(EquityAgent(seed=0))['members']
    print('    TAG :', {k: f"{v['mean']:+.0f}±{v['se']:.0f}" for k, v in tag.items()})
    assert tag['nit']['mean'] > 3 * tag['nit']['se'], tag['nit']
    assert abs(tag['tag']['mean']) < 4 * tag['tag']['se'] + 5, tag['tag']
    # The lag is the strongest style; its gap to the tag is small (~+80
    # mbb/h), so that comparison gets its own, larger match.
    lag = Panel(members=('lag',), num_deals=5000).evaluate(EquityAgent(seed=0))['members']['lag']
    print(f"    TAG vs LAG over 5000 deals: {lag['mean']:+.0f}±{lag['se']:.0f}")
    assert lag['mean'] > 3 * lag['se'], lag
    for name in ('nit', 'lag'):
        r = panel.evaluate(MEMBERS[name]())['members']
        print(f'    {name.upper()} :', {k: f"{v['mean']:+.0f}±{v['se']:.0f}" for k, v in r.items()})
        for trivial in ('random', 'station', 'maniac'):
            assert r[trivial]['mean'] > 3 * r[trivial]['se'], (name, trivial, r[trivial])


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
