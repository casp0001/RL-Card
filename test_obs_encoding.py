''' Tests for obs_encoding.py (the rich observation).   Run:  python test_obs_encoding.py [-k name] '''

import os
import sys
import time
import random
import argparse
import tempfile
import traceback
import contextlib

import numpy as np

import rlcard_patches  # noqa: F401
import rlcard
from obs_encoding import RICH_DIM, CARD_INDEX, ENV_IDS, encoding_of_dim

CALL, RAISE, FOLD, CHECK = 0, 1, 2, 3


def random_states(env_id='limit-holdem', n=3000, seed=0):
    ''' (state, env-snapshot info) pairs from random play. '''
    env = rlcard.make(env_id, config={'seed': seed})
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        state, player = env.reset()
        sb = next(i for i, p in enumerate(env.game.players) if p.in_chips == 1)
        actions = []                                      # (round, player, action) so far
        while not env.is_over():
            # raw_obs['raise_nums'] is the game's live list: snapshot it now
            state = dict(state, raw_obs=dict(state['raw_obs'], raise_nums=list(state['raw_obs']['raise_nums'])))
            out.append((state, player, sb, list(actions), env.game.round_counter))
            a = rng.choice(list(state['legal_actions']))
            actions.append((env.game.round_counter, player, env.actions[a]))
            state, player = env.step(a)
    return out


def test_sections_match_the_game():
    for state, player, sb, actions, rnd in random_states():
        o = state['obs_rich']
        raw = state['raw_obs']
        assert o.shape == (RICH_DIM,)
        assert sorted(np.flatnonzero(o[:52])) == sorted(CARD_INDEX[c] for c in raw['hand'])
        assert sorted(np.flatnonzero(o[52:104])) == sorted(CARD_INDEX[c] for c in raw['public_cards'])
        assert np.flatnonzero(o[104:108]).tolist() == [{0: 0, 3: 1, 4: 2, 5: 3}[len(raw['public_cards'])]]
        assert np.flatnonzero(o[108:110]).tolist() == [0 if player == sb else 1]
        to_call = max(raw['all_chips']) - raw['my_chips']
        assert o[110] == (1 if to_call > 0 else 0) and abs(o[152] - to_call / 4) < 1e-12
        for i, n in enumerate(raw['raise_nums']):
            assert np.flatnonzero(o[111 + 5 * i:116 + 5 * i]).tolist() == [n]
        this = [(p, a) for r, p, a in actions if r == rnd]
        mine = sum(1 for p, a in this if p == player and a == 'raise')
        theirs = sum(1 for p, a in this if p != player and a == 'raise')
        assert np.flatnonzero(o[131:136]).tolist() == [mine] and np.flatnonzero(o[136:141]).tolist() == [theirs]
        assert mine + theirs == raw['raise_nums'][rnd]
        assert o[141] == (1 if any(p == player for p, _ in this) else 0)
        for r in range(3):
            block = o[142 + 3 * r:145 + 3 * r]
            if r >= rnd:
                assert not block.any()
                continue
            raisers = [p for rr, p, a in actions if rr == r and a == 'raise']
            slot = 0 if not raisers else (1 if raisers[-1] == player else 2)
            assert np.flatnonzero(block).tolist() == [slot]
        assert abs(o[151] - sum(raw['all_chips']) / 100) < 1e-12 and 0 <= o[151] <= 1


def test_rich_env_and_default_obs():
    ''' The rich env's obs is the rich observation; the default env keeps
    RLCard's 72-value obs and adds obs_rich; both see the same game. '''
    a = rlcard.make('limit-holdem-rich', config={'seed': 3})
    b = rlcard.make('limit-holdem', config={'seed': 3})
    assert a.state_shape == [[RICH_DIM], [RICH_DIM]] and b.state_shape == [[72], [72]]
    rng = random.Random(3)
    for _ in range(200):
        sa, _ = a.reset()
        sb, _ = b.reset()
        while not a.is_over():
            assert np.array_equal(sa['obs'], sb['obs_rich']) and sb['obs'].shape == (72,)
            assert np.array_equal(sa['obs_rlcard'], sb['obs']) and np.array_equal(sa['obs_rich'], sb['obs_rich'])
            act = rng.choice(list(sa['legal_actions']))
            sa, _ = a.step(act)
            sb, _ = b.step(act)


def test_step_back_restores_the_observation():
    env = rlcard.make('limit-holdem', config={'seed': 4, 'allow_step_back': True})
    rng = random.Random(4)
    for _ in range(300):
        state, player = env.reset()
        stack = []
        for _ in range(30):
            if env.is_over() or (stack and rng.random() < 0.4):
                if not stack:
                    break
                restored, _ = env.step_back()
                expected = stack.pop()
                assert np.array_equal(restored['obs_rich'], expected), 'step_back changed obs_rich'
                state = restored
            else:
                stack.append(state['obs_rich'].copy())
                state, player = env.step(rng.choice(list(state['legal_actions'])))


def test_formerly_aliased_situations_differ():
    ''' Two situations RLCard's 72-value obs could not tell apart. '''
    env = rlcard.make('limit-holdem', config={'seed': 0})
    for seed in range(200):
        env.seed(seed)
        sb_state, _ = env.reset()                  # small blind, first to act
        bb_state, _ = env.step(CALL)                # big blind after the limp
        if env.is_over():
            continue
        cards_sb = sb_state['raw_obs']['hand']
        cards_bb = bb_state['raw_obs']['hand']
        if sorted(cards_sb) != sorted(cards_bb):
            # compare the betting parts only (cards differ by construction)
            assert np.array_equal(sb_state['obs'][52:], bb_state['obs'][52:])     # aliased in rlcard obs
            assert not np.array_equal(sb_state['obs_rich'][104:], bb_state['obs_rich'][104:])
            break
    # a pocket pair vs a pair on the board: same 52-bit card set in RLCard's obs
    from obs_encoding import encode_rich

    class G(object):
        sb_seat, round_counter, action_history = 0, 1, []
    raw = lambda hand, board: {'player_id': 0, 'hand': hand, 'public_cards': board,
                               'all_chips': [2, 2], 'my_chips': 2, 'raise_nums': [0, 0, 0, 0]}
    pocket = encode_rich(G, raw(['SA', 'HA'], ['D2', 'C7', 'S9']))
    board = encode_rich(G, raw(['D2', 'C7'], ['SA', 'HA', 'S9']))
    assert not np.array_equal(pocket, board)
    assert np.array_equal(pocket[:52] + pocket[52:104], board[:52] + board[52:104])   # same cards overall


def test_training_and_loading_both_encodings():
    ''' Agents train on either encoding and the loader picks the right one;
    a rich agent's policy probabilities in a default env equal its own
    probabilities on the rich env's states. '''
    import importlib
    from agent_loading import load_policy
    with tempfile.TemporaryDirectory() as d:
        for obs in ('rich', 'rlcard'):
            for name, kind, extra in (
                    ('limit_dqn_selfplay_custom', 'dqn', {'replay_memory_init_size': 100}),
                    ('limit_ppo_selfplay', 'ppo', {'update_every': 256}),
                    ('limit_nfsp_selfplay_custom_hyperparams', 'nfsp', {
                        'hidden_layers_sizes': [32], 'q_mlp_layers': [32], 'reservoir_buffer_capacity': 5000,
                        'q_replay_memory_size': 5000, 'min_buffer_size_to_learn': 300,
                        'q_replay_memory_init_size': 300, 'train_every': 32, 'q_train_every': 32}),
                    ('limit_deepcfr_train_custom', 'deepcfr', {
                        'num_episodes': 2, 'evaluate_every': 1, 'num_traversals': 5, 'num_step': 3,
                        'batch_size_advantage': 32, 'batch_size_strategy': 32, 'policy_layers': [32],
                        'advantage_layers': [32]})):
                module = importlib.import_module(name)
                args = module.build_parser().parse_args([])
                log_dir = os.path.join(d, f'{kind}_{obs}') + os.sep
                settings = {'num_episodes': 400, 'evaluate_every': 400, 'opponent_update_every': 100,
                            'eval_deals': 3, 'verbose': False, 'obs': obs, 'log_dir': log_dir}
                for k, v in {**settings, **extra}.items():
                    setattr(args, k, v)
                with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
                    module.train(args)
                policy = load_policy(kind, log_dir)
                assert policy.encoding == obs, (kind, obs, policy.encoding)
                # same probabilities from the default env's obs_rich / obs as from the agent's own env
                for env_id in ('limit-holdem', 'limit-holdem-rich'):
                    env = rlcard.make(env_id, config={'seed': 9})
                    state, _ = env.reset()
                    p = policy.probs(state)
                    assert abs(p.sum() - 1) < 1e-6
                    own_env = rlcard.make(ENV_IDS[obs], config={'seed': 9})
                    own_state, _ = own_env.reset()
                    obs_vec = np.asarray(own_state['obs'], dtype=np.float32)[None]
                    assert np.allclose(p, policy.probs_batch_obs(obs_vec, list(own_state['legal_actions']))[0])
                print(f'    {kind:8s} {obs:6s}: trained, loaded as {policy.encoding}')


def test_encoding_of_dim():
    assert encoding_of_dim(72) == 'rlcard' and encoding_of_dim(RICH_DIM) == 'rich'
    try:
        encoding_of_dim(100)
    except ValueError:
        return
    raise AssertionError('unknown size accepted')


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
