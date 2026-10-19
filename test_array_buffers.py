''' Tests for array_buffers.py and the agents / helpers that use it.
Run:  python test_array_buffers.py [-k name]
'''

import os
import sys
import math
import time
import random
import argparse
import tempfile
import traceback
from collections import Counter, namedtuple

import numpy as np
import torch

import rlcard_patches  # noqa: F401
import rlcard
from rlcard.utils import reorganize

from array_buffers import Reservoir, Ring, ReplayMemory, NullBuffer


def test_reservoir_is_uniform():
    ''' Every item ever added is kept with probability capacity / n. '''
    capacity, n, runs = 100, 5000, 400
    kept = Counter()
    for run in range(runs):
        random.seed(run)
        r = Reservoir(capacity, {'x': ((), np.int64)}, initial=8)
        for i in range(n):
            r.add({'x': i})
        assert len(r) == capacity and r.add_calls == n
        for v in r.arrays()['x']:
            kept[int(v) * 10 // n] += 1                    # decile of the item's arrival time
    expected = runs * capacity / 10
    for decile in range(10):
        assert abs(kept[decile] - expected) < 5 * math.sqrt(expected), (decile, kept[decile], expected)


def test_ring_keeps_newest():
    r = Ring(5, {'x': ((), np.int64)}, initial=2)
    for i in range(12):
        r.add({'x': i})
    assert sorted(r.arrays()['x'].tolist()) == [7, 8, 9, 10, 11]


def test_sampling_and_growth():
    r = Reservoir(10_000_000, {'obs': ((72,), np.float16)}, initial=16)
    assert r.data['obs'].nbytes < 10_000                    # no up-front allocation
    for i in range(1000):
        r.add({'obs': np.full(72, i % 7)})
    random.seed(1)
    batch = r.sample(300)['obs']
    assert batch.shape == (300, 72) and set(np.unique(batch)) <= set(range(7))
    try:
        r.sample(2000)
    except ValueError:
        pass
    else:
        raise AssertionError('oversampling accepted')
    # checkpoint round trip
    state = r.checkpoint_attributes()
    back = Reservoir.from_state(state)
    assert len(back) == 1000 and np.array_equal(back.arrays()['obs'], r.arrays()['obs'])
    assert back.add_calls == r.add_calls
    empty = Reservoir.from_state(r.checkpoint_attributes(include_data=False))
    assert len(empty) == 0 and empty.capacity == r.capacity


def _transitions(n, seed=0):
    env = rlcard.make('limit-holdem', config={'seed': seed})
    from rlcard.agents import RandomAgent
    env.set_agents([RandomAgent(4), RandomAgent(4)])
    out = []
    while len(out) < n:
        trajectories, payoffs = env.run(is_training=False)
        for seat_ts in reorganize(trajectories, payoffs):
            out.extend(seat_ts)
    return out[:n]


def test_replay_memory():
    ts = _transitions(500)
    mem = ReplayMemory(300, 32, num_actions=4)
    for s, a, r, ns, d in ts:
        mem.save(s['obs'], a, r, ns['obs'], list(ns['legal_actions']), d)
    assert len(mem) == 300 and len(mem.memory) == 300
    random.seed(0)
    states, actions, rewards, next_states, dones, legal = mem.sample()
    assert states.dtype == np.float32 and states.shape == (32, 72) and legal.shape == (32, 4)
    # every sampled row is one of the newest 300 transitions, with its own legal mask
    newest = {(tuple(s['obs']), a, float(r), tuple(ns['obs']), d): tuple(sorted(ns['legal_actions']))
              for s, a, r, ns, d in ts[-300:]}
    for i in range(32):
        key = (tuple(states[i]), int(actions[i]), float(rewards[i]), tuple(next_states[i]), bool(dones[i]))
        assert key in newest, i
        assert tuple(np.flatnonzero(legal[i])) == newest[key]
    # checkpoints: new format, without data, and the old list-of-Transition format
    back = ReplayMemory.from_checkpoint(mem.checkpoint_attributes())
    assert len(back) == 300 and np.array_equal(back.ring.arrays()['state'], mem.ring.arrays()['state'])
    assert len(ReplayMemory.from_checkpoint(mem.checkpoint_attributes(include_data=False))) == 0
    T = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])
    old = {'memory_size': 300, 'batch_size': 32,
           'memory': [T(s['obs'], a, r, ns['obs'], d, list(ns['legal_actions'])) for s, a, r, ns, d in ts[:50]]}
    assert len(ReplayMemory.from_checkpoint(old)) == 50


def test_memory_per_sample():
    ''' Bytes per stored sample, vs ~1576 (DQN), ~913 (NFSP), ~785 (Deep CFR) before. '''
    from deep_cfr_agent_new import DeepCFR
    from nfsp_agent_XPU import ReservoirBuffer, Transition
    ts = _transitions(2000)
    mem = ReplayMemory(10_000, 32)
    for s, a, r, ns, d in ts:
        mem.save(s['obs'], a, r, ns['obs'], list(ns['legal_actions']), d)
    dqn = mem.ring.nbytes() / len(mem)
    res = ReservoirBuffer(10_000)
    for s, a, r, ns, d in ts:
        res.add(Transition(s['obs'], np.eye(4)[a]))
    nfsp = res.nbytes() / len(res)
    env = rlcard.make('limit-holdem', config={'allow_step_back': True})
    cfr = DeepCFR(env=env, memory_capacity=10_000)
    for s, a, r, ns, d in ts:
        cfr._advantage_memories[0].add({'info_state': s['obs'], 'iteration': 1, 'advantage': 0.5, 'action': a})
    adv = cfr._advantage_memories[0].nbytes() / len(cfr._advantage_memories[0])
    print(f'    bytes per sample: DQN {dqn:.0f} (was 1576), NFSP {nfsp:.0f} (was 913), Deep CFR {adv:.0f} (was 785)')
    assert dqn < 400 and nfsp < 200 and adv < 200


def test_dqn_trains_and_checkpoints():
    from dqn_agent_XPU import DQNAgent
    agent = DQNAgent(num_actions=4, state_shape=[72], mlp_layers=[32], replay_memory_size=2000,
                     replay_memory_init_size=100, batch_size=32, discount_factor=1.0, device='cpu')
    for ts in _transitions(800):
        agent.feed(ts)
    assert agent.train_t > 0
    with tempfile.TemporaryDirectory() as d:
        agent.save_checkpoint(d)                               # light by default
        light = torch.load(os.path.join(d, 'checkpoint_dqn.pt'), weights_only=False)
        agent.save_checkpoint(d, 'full.pt', include_buffers=True)
        full = torch.load(os.path.join(d, 'full.pt'), weights_only=False)
        assert os.path.getsize(os.path.join(d, 'checkpoint_dqn.pt')) < os.path.getsize(os.path.join(d, 'full.pt'))
    a2 = DQNAgent.from_checkpoint(full)
    assert len(a2.memory) == len(agent.memory)
    a3 = DQNAgent.from_checkpoint(light)
    assert len(a3.memory) == 0
    s = _transitions(5)[0][0]
    assert a2.eval_step(s)[0] == a3.eval_step(s)[0] == agent.eval_step(s)[0]


def test_old_checkpoints_still_load():
    ''' Checkpoints written before the array buffers (if the exploitability
    tests' tiny checkpoints exist from an earlier run) still load. '''
    base = os.path.join(tempfile.gettempdir(), 'test_exploitability_ckpts')
    paths = {'dqn': os.path.join(base, 'dqn', 'checkpoint_dqn.pt'),
             'nfsp': os.path.join(base, 'nfsp', 'checkpoint_nfsp.pt')}
    if not all(os.path.isfile(p) for p in paths.values()):
        print('    (skipped: no old checkpoints)')
        return
    from dqn_agent_XPU import DQNAgent
    from nfsp_agent_XPU import NFSPAgent
    from agent_loading import load_policy
    dqn = DQNAgent.from_checkpoint(torch.load(paths['dqn'], weights_only=False))
    nfsp = NFSPAgent.from_checkpoint(torch.load(paths['nfsp'], weights_only=False))
    print(f'    loaded old dqn ({len(dqn.memory)} transitions) and nfsp '
          f'({len(nfsp._reservoir_buffer)} reservoir entries)')
    for kind in paths:
        load_policy(kind, os.path.dirname(paths[kind]))


def test_frozen_copy():
    from functions import frozen_copy
    from dqn_agent_XPU import DQNAgent
    from nfsp_agent_XPU import NFSPAgent
    from ppo_agent import PPOAgent
    ts = _transitions(400)
    dqn = DQNAgent(num_actions=4, state_shape=[72], mlp_layers=[16], replay_memory_size=1000,
                   replay_memory_init_size=10**9, device='cpu')
    for t in ts:
        dqn.feed(t)
    opp = frozen_copy(dqn)
    assert isinstance(opp.memory, NullBuffer) and len(dqn.memory) == 400
    s = ts[0][0]
    assert opp.eval_step(s)[0] == dqn.eval_step(s)[0]
    opp.step(s)

    nfsp = NFSPAgent(num_actions=4, state_shape=[72], hidden_layers_sizes=[16], q_mlp_layers=[16],
                     reservoir_buffer_capacity=1000, min_buffer_size_to_learn=10**9,
                     q_replay_memory_init_size=10**9, device='cpu')
    for t in ts:
        nfsp._mode = 'best_response'
        nfsp.step(t[0])
        nfsp.feed(t)
    opp = frozen_copy(nfsp)
    assert isinstance(opp._reservoir_buffer, NullBuffer) and isinstance(opp._rl_agent.memory, NullBuffer)
    assert len(nfsp._reservoir_buffer) > 0 and len(nfsp._rl_agent.memory) == 400
    for mode in ('best_response', 'average_policy'):
        opp._mode = mode
        opp.step(s)                                          # adds to the NullBuffer: no error
    assert np.allclose(opp._act(s['obs']), nfsp._act(s['obs']))

    ppo = PPOAgent(num_actions=4, state_shape=[72], mlp_layers=[16], device='cpu')
    opp = frozen_copy(ppo)
    np.random.seed(0)
    picks = Counter(opp.step(s) for _ in range(3000))
    assert opp._pending == []                               # nothing queued for training
    log_probs, _ = ppo._policy_nograd(np.asarray(s['obs'], dtype=np.float32), list(s['legal_actions']))
    probs = np.exp(log_probs)
    for a in s['legal_actions']:
        assert abs(picks[a] / 3000 - probs[a]) < 0.04, (a, picks, probs)


def test_scripts_still_train():
    ''' The training scripts run end to end with the new buffers / snapshots. '''
    import importlib
    import contextlib
    jobs = {
        'limit_dqn_selfplay_custom': {'num_episodes': 600, 'evaluate_every': 300, 'opponent_update_every': 100,
                                      'replay_memory_init_size': 100, 'checkpoint_every': 300},
        'limit_nfsp_selfplay_custom_hyperparams': {
            'num_episodes': 600, 'evaluate_every': 300, 'opponent_update_every': 100, 'hidden_layers_sizes': [32],
            'q_mlp_layers': [32], 'reservoir_buffer_capacity': 5000, 'q_replay_memory_size': 5000,
            'min_buffer_size_to_learn': 300, 'q_replay_memory_init_size': 300, 'train_every': 32,
            'q_train_every': 32, 'checkpoint_every': 300},
        'limit_ppo_selfplay': {'num_episodes': 600, 'evaluate_every': 300, 'opponent_update_every': 100,
                               'update_every': 256},
        'limit_deepcfr_train_custom': {'num_episodes': 3, 'evaluate_every': 1, 'num_traversals': 10,
                                       'num_step': 5, 'batch_size_advantage': 32, 'batch_size_strategy': 32,
                                       'policy_layers': [32], 'advantage_layers': [32], 'checkpoint_every': 2},
    }
    with tempfile.TemporaryDirectory() as d:
        for name, overrides in jobs.items():
            module = importlib.import_module(name)
            args = module.build_parser().parse_args([])
            for k, v in {**overrides, 'eval_deals': 5, 'verbose': False,
                         'log_dir': os.path.join(d, name) + os.sep}.items():
                setattr(args, k, v)
            history = []
            with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
                agent = module.train(args, lambda ep, r: history.append(r['min']) and False)
            assert history, name
            print(f'    {name}: ok ({len(history)} evaluations)')


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
