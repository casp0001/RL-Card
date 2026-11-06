''' Tests for leduc.py (exact best response and tabular CFR on Leduc).
Run:  python test_leduc.py  [-k name]     (about 2 minutes)
'''

import sys
import math
import time
import argparse
import traceback

import numpy as np
import rlcard

import leduc as L


def _play_env(env, policies, seed):
    ''' One hand in RLCard's env with policies by position (0 = small blind).
    Replays LeducState alongside and checks it against the env at every
    step. Returns the small blind's payoff. '''
    rng = np.random.RandomState(seed)
    _, pid = env.reset()
    sb = pid                                     # the small blind acts first
    rank = {i: L.RANK_NAMES.index(env.game.players[i].hand.rank) for i in (0, 1)}
    model = L.LeducState()
    while not env.is_over():
        pos = 0 if pid == sb else 1
        assert model.pointer == pos and not model.is_terminal()
        public = L.RANK_NAMES.index(env.game.public_card.rank) if env.game.public_card else None
        info = model.info(rank[pid], public)
        state = env.get_state(pid)
        assert list(state['legal_actions']) == list(info.state['legal_actions']), (state['raw_legal_actions'], model.history)
        assert np.array_equal(state['obs'], info.state['obs']), model.history
        p = np.asarray(policies[pos](info), dtype=np.float64)
        a = int(rng.choice(L.NUM_ACTIONS, p=p / p.sum()))
        model = model.child(a)
        _, pid = env.step(a)
    assert model.is_terminal()
    public = L.RANK_NAMES.index(env.game.public_card.rank) if env.game.public_card else None
    payoffs = env.get_payoffs()
    ranks = (rank[sb], rank[1 - sb])
    # a hand that ends by a fold in round 1 never deals the public card; the payoff does not depend on it
    expected = model.payoff(0, ranks, public if public is not None else -1)
    assert math.isclose(payoffs[sb], expected), (payoffs, expected, model.history)
    return payoffs[sb]


def _obs_policy(seed):
    ''' A fixed random policy of the observation only (as a network is). '''
    def policy(info):
        legal = list(info.state['legal_actions'])
        h = abs(hash((seed, info.state['obs'].tobytes(), tuple(legal)))) % (2 ** 32)
        p = np.zeros(L.NUM_ACTIONS)
        p[legal] = np.random.RandomState(h).dirichlet(np.ones(len(legal)))
        return p
    return policy


def _strategy_policy(strategy, seat):
    def policy(info):
        rank, public, history = info.key
        p = np.zeros(L.NUM_ACTIONS)
        p[strategy[(seat, rank, public, history)]] = 1.0
        return p
    return policy


def test_state_matches_rlcard():
    env = rlcard.make('leduc-holdem', config={'seed': 0})
    for i in range(3000):
        _play_env(env, (L.uniform_policy, L.uniform_policy), i)


def test_uniform_is_symmetric():
    assert abs(L.match_value(L.uniform_policy, L.uniform_policy)) < 1e-12
    pol = _obs_policy(1)
    assert abs(L.match_value(pol, pol)) < 1e-12


def test_best_response_value_is_achieved():
    ''' The BR's own strategy, evaluated exactly, earns the reported value. '''
    for pol, by_key in ((L.uniform_policy, False), (_obs_policy(2), False),
                        (L.TabularCFR().iterate(30).average_policy(), True)):
        br = L.best_response(pol, by_key)
        for seat, name in ((0, 'small_blind'), (1, 'big_blind')):
            mine = L._Table(_strategy_policy(br['strategy'], seat), True)
            other = L._Table(pol, by_key)
            tables = (mine, other) if seat == 0 else (other, mine)
            v = sum(p * L._match_value(L.LeducState(), (a, b), c, tables) for a, b, c, p in L._deals())
            v = v if seat == 0 else -v
            assert math.isclose(v, br[name], abs_tol=1e-12), (name, v, br[name])


def test_best_response_beats_other_strategies():
    ''' No other policy (random ones, CFR) does better against the target. '''
    target = _obs_policy(3)
    expl = L.exploitability(target)
    for s in range(20):
        assert L.match_value(_obs_policy(100 + s), target) <= expl + 1e-12
    assert L.match_value(L.TabularCFR().iterate(200).average_policy(), target, a_by_key=True) <= expl + 1e-12


def test_monte_carlo_in_env():
    ''' The BR's value, played out in RLCard's env, matches the exact value. '''
    target = _obs_policy(4)
    br = L.best_response(target)
    env = rlcard.make('leduc-holdem', config={'seed': 1})
    for seat, name in ((0, 'small_blind'), (1, 'big_blind')):
        mine = _strategy_policy(br['strategy'], seat)
        pols = (mine, target) if seat == 0 else (target, mine)
        n = 20000
        x = np.array([_play_env(env, pols, 10 ** 6 * (seat + 1) + i) for i in range(n)])
        x = x if seat == 0 else -x
        se = x.std() / math.sqrt(n)
        assert abs(x.mean() - br[name]) < 4 * se, (name, x.mean(), br[name], se)


def test_cfr_converges():
    cfr = L.TabularCFR().iterate(1000)
    assert L.exploitability(cfr.average_policy(), True) < 0.2e-3
    assert L.exploitability(L.uniform_policy) > 1.0


def test_exploitability_nonnegative():
    for s in range(10):
        assert L.exploitability(_obs_policy(200 + s)) > 0


def test_trained_agents_match_eval_step():
    ''' policy_of gives the distribution each training agent's eval_step plays:
    the exact value against the uniform policy matches play in the env. '''
    import os
    import tempfile
    import contextlib
    import importlib
    import torch
    from eval_panel import eval_mode
    jobs = {'dqn': ('limit_dqn_selfplay_custom', {'num_episodes': 1500, 'replay_memory_init_size': 200}),
            'ppo': ('limit_ppo_selfplay', {'num_episodes': 1500, 'update_every': 512}),
            'nfsp': ('limit_nfsp_selfplay_custom_hyperparams', {'num_episodes': 1500, 'min_buffer_size_to_learn': 200}),
            'deepcfr': ('limit_deepcfr_train_custom', {'num_episodes': 2, 'num_traversals': 20})}
    env = rlcard.make('leduc-holdem', config={'seed': 5})
    for kind, (module_name, overrides) in jobs.items():
        module = importlib.import_module(module_name)
        args = module.build_parser().parse_args([])
        for k, v in dict(overrides, game='leduc', verbose=False, save_model=False,
                         evaluate_every=10 ** 9, log_dir=tempfile.mkdtemp() + os.sep).items():
            setattr(args, k, v)
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            agent = module.train(args)
        if kind == 'deepcfr' and not isinstance(agent, module.DeepCFREvalAgent):
            agent = module.DeepCFREvalAgent(agent)
        exact = L.match_value(L.policy_of(agent), L.uniform_policy)
        uniform = rlcard.agents.RandomAgent(num_actions=4)
        n, x = 10000, []
        with eval_mode(agent):
            for i in range(n):
                env.seed(i)
                np.random.seed(i)
                torch.manual_seed(i)
                seat = i % 2
                env.set_agents([agent, uniform] if seat == 0 else [uniform, agent])
                _, payoffs = env.run(is_training=False)
                x.append(payoffs[seat])
        x = np.array(x)
        se = x.std() / math.sqrt(n)
        assert abs(x.mean() - exact) < 4 * se, (kind, x.mean(), exact, se)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', default='')
    a = parser.parse_args()
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith('test_') and callable(f) and a.k in n]
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
