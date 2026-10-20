''' Export agents and the poker engine to the web app repository (RL-Card-web).

    python export_for_web.py --web_repo ../RL-Card-web \
        --agent dqn=dqn:experiments/final/.../seed_1042 \
        --agent nfsp=nfsp:experiments/tune/nfsp_v1/finalists/trial_3/seed_1042 ...
    python export_for_web.py --web_repo ../RL-Card-web --from_manifest experiments/final/manifest.json

For every trained agent it
  1. converts the network to numpy operations (numpy_policies.py) and writes
     agents/<name>.npz,
  2. checks on 2000 real game states that the numpy policy gives the same
     action probabilities as the torch policy (agent_loading.py), and refuses
     to export otherwise,
  3. writes agents/<name>.check.npz (sample observations + expected
     probabilities) so the web repo can re-check the numpy agents without torch.
Rule-based bots (tag, nit, lag, honest, station, maniac, random) need no files;
their settings go into agents/agents.json.

It also copies the engine modules the web app needs into engine/ (stamped
with this repository's commit in engine/VERSION.json).

--from_manifest takes the first seed of every algorithm in a
final_evaluation.py manifest (names: dqn, ppo, nfsp, deepcfr).
'''

import os
import sys
import json
import time
import shutil
import hashlib
import argparse
import subprocess

import numpy as np
import torch
import rlcard

import rlcard_patches  # noqa: F401
from agent_loading import load_policy, CHECKPOINT_KINDS
from numpy_policies import NumpyPolicy

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ENGINE_FILES = ['rlcard_patches.py', 'obs_encoding.py', 'equity_agent.py', 'preflop_equity.json',
                'behaviour_tracker.py', 'honest_agent_limited.py', 'numpy_policies.py']
BOT_DESCRIPTIONS = {
    'tag': 'Rule-based: tight-aggressive, plays by hand equity and pot odds',
    'nit': 'Rule-based: tight-passive, folds a lot, never bluffs',
    'lag': 'Rule-based: loose-aggressive, raises and bluffs a lot',
    'honest': 'Rule-based: the simple first bot of the thesis (weak)',
    'station': 'Always checks or calls',
    'maniac': 'Always raises',
    'random': 'Plays uniformly at random',
}
ALGO_DESCRIPTIONS = {
    'dqn': 'Deep Q-Network (self-play)',
    'ppo': 'Proximal Policy Optimization (self-play)',
    'nfsp': 'Neural Fictitious Self-Play',
    'deepcfr': 'Deep Counterfactual Regret Minimization',
}


# --------------------------------------------------------------------------- #
# torch network -> numpy operations
# --------------------------------------------------------------------------- #

def _ops_of(modules):
    ''' (meta, arrays) for a sequence of torch modules. '''
    import torch.nn as nn
    ops, arrays = [], {}
    for m in modules:
        i = len(ops)
        if isinstance(m, nn.Flatten):
            continue
        if isinstance(m, nn.BatchNorm1d):
            ops.append({'type': 'batchnorm', 'arrays': ['mean', 'var', 'weight', 'bias'], 'eps': m.eps})
            arrays.update({f'op{i}_mean': m.running_mean, f'op{i}_var': m.running_var,
                           f'op{i}_weight': m.weight, f'op{i}_bias': m.bias})
        elif isinstance(m, nn.Linear):
            ops.append({'type': 'linear', 'arrays': ['W', 'b']})
            arrays.update({f'op{i}_W': m.weight, f'op{i}_b': m.bias})
        elif isinstance(m, nn.ReLU):
            ops.append({'type': 'relu'})
        elif isinstance(m, nn.Tanh):
            ops.append({'type': 'tanh'})
        else:
            raise ValueError(f'cannot export module {type(m).__name__}')
    return ops, {k: v.detach().cpu().numpy() for k, v in arrays.items()}


def export_network(policy):
    ''' (ops meta, arrays, output mode, input dim) of an agent_loading Policy. '''
    kind = policy.kind
    if kind == 'dqn':
        modules, output = list(policy.agent.q_estimator.qnet.fc_layers), 'greedy_q'
    elif kind == 'nfsp':
        # the network applies log_softmax after `mlp`; exp(log_softmax), restricted
        # to the legal actions and renormalised, equals softmax_renorm of the logits
        modules, output = list(policy.agent.policy_network.mlp), 'softmax_renorm'
    elif kind == 'deepcfr':
        modules, output = list(policy.agent._policy_network), 'softmax_renorm'
    elif kind == 'ppo':
        modules = list(policy.agent.network.trunk) + [policy.agent.network.actor_head]
        output = 'masked_greedy' if policy.agent.evaluate_with == 'greedy' else 'masked_softmax'
    else:
        raise ValueError(kind)
    ops, arrays = _ops_of(modules)
    return ops, arrays, output, policy.encoding


def write_policy(path, name, kind, ops, arrays, output, encoding, input_dim):
    meta = {'name': name, 'kind': kind, 'encoding': encoding, 'output': output, 'ops': ops,
            'input_dim': int(input_dim), 'exported': time.strftime('%Y-%m-%d %H:%M:%S')}
    np.savez_compressed(path, __meta__=np.array(json.dumps(meta)), **arrays)


def sample_states(n=2000, seed=123):
    import random
    env = rlcard.make('limit-holdem', config={'seed': seed})
    rng = random.Random(seed)
    states = []
    while len(states) < n:
        state, _ = env.reset()
        while not env.is_over():
            states.append(state)
            state, _ = env.step(rng.choice(list(state['legal_actions'])))
    return states[:n]


def verify(torch_policy, numpy_policy, states):
    ''' Largest probability difference, and the number of greedy ties that
    may legitimately flip between float32 (torch) and float64 (numpy). '''
    worst, flips = 0.0, 0
    for s in states:
        a = np.asarray(torch_policy.probs(s), dtype=np.float64)
        b = numpy_policy.probs(s)
        diff = float(np.abs(a - b).max())
        if numpy_policy.output in ('greedy_q', 'masked_greedy') and diff > 0:
            # a flip is acceptable only if the two best values are (nearly) tied
            obs = np.asarray(s[numpy_policy.obs_key], dtype=np.float64)[None]
            values = numpy_policy.forward(obs)[0][list(s['legal_actions'])]
            top = np.sort(values)[-2:]
            if len(top) == 2 and abs(top[1] - top[0]) < 1e-4:
                flips += 1
                continue
        worst = max(worst, diff)
    return worst, flips


def git_commit():
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, capture_output=True,
                                text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=REPO_ROOT, capture_output=True,
                                    text=True).stdout.strip())
        return commit, dirty
    except OSError:
        return None, None


def _repo_path(ckpt):
    ''' The checkpoint relative to the thesis repo; just its folder name when it
    lies outside (the web repo is public, local paths do not belong there). '''
    if not ckpt:
        return None
    try:
        rel = os.path.relpath(os.path.abspath(ckpt), REPO_ROOT)
    except ValueError:              # another drive
        rel = '..'
    return rel.replace('\\', '/') if not rel.startswith('..') else os.path.basename(os.path.normpath(ckpt))


def main():
    p = argparse.ArgumentParser('Export agents and the engine to the web app repository')
    p.add_argument('--web_repo', default=os.path.join(os.path.dirname(REPO_ROOT), 'RL-Card-web'))
    p.add_argument('--agent', action='append', default=[],
                   help='name=kind:checkpoint (trained) -- repeatable')
    p.add_argument('--from_manifest', default=None, help='final_evaluation.py manifest: first seed per algorithm')
    p.add_argument('--bots', default='tag,nit,lag,honest,station,maniac,random')
    p.add_argument('--random_pool', default=None,
                   help='comma-separated agent names the "mystery opponent" is drawn from '
                        '(default: all trained agents + tag, nit, lag)')
    p.add_argument('--engine_only', action='store_true', help='only copy the engine files')
    a = p.parse_args()

    web = os.path.abspath(a.web_repo)
    os.makedirs(os.path.join(web, 'engine'), exist_ok=True)
    os.makedirs(os.path.join(web, 'agents'), exist_ok=True)

    # 1. engine snapshot
    digests = {}
    for f in ENGINE_FILES:
        shutil.copy2(os.path.join(REPO_ROOT, f), os.path.join(web, 'engine', f))
        with open(os.path.join(REPO_ROOT, f), 'rb') as h:
            digests[f] = hashlib.sha256(h.read()).hexdigest()[:16]
    commit, dirty = git_commit()
    with open(os.path.join(web, 'engine', 'VERSION.json'), 'w', encoding='utf-8') as f:
        json.dump({'thesis_commit': commit, 'uncommitted_changes': dirty,
                   'exported': time.strftime('%Y-%m-%d %H:%M:%S'), 'files': digests}, f, indent=2)
    print(f'engine: {len(ENGINE_FILES)} files -> {web}/engine (thesis commit {commit}{" + local changes" if dirty else ""})')
    if a.engine_only:
        return

    # 2. agents
    specs = []
    for spec in a.agent:
        name, rest = spec.split('=', 1)
        kind, _, ckpt = rest.partition(':')
        specs.append((name, kind, ckpt or None))
    if a.from_manifest:
        with open(a.from_manifest, encoding='utf-8') as f:
            seen = set()
            for e in json.load(f):
                if e['kind'] in CHECKPOINT_KINDS and e['algo'] not in seen:
                    seen.add(e['algo'])
                    ckpt = e['ckpt'] if os.path.isabs(e['ckpt']) else os.path.join(REPO_ROOT, e['ckpt'])
                    specs.append((e['algo'], e['kind'], ckpt))
    states = sample_states() if specs else []
    entries = []
    for name, kind, ckpt in specs:
        if kind not in CHECKPOINT_KINDS:
            raise SystemExit(f'{name}: --agent is for trained agents ({CHECKPOINT_KINDS}); bots go in --bots')
        torch_policy = load_policy(kind, ckpt)
        ops, arrays, output, encoding = export_network(torch_policy)
        input_dim = np.asarray(states[0][torch_policy.obs_key]).size
        path = os.path.join(web, 'agents', f'{name}.npz')
        write_policy(path, name, kind, ops, arrays, output, encoding, input_dim)
        numpy_policy = NumpyPolicy(path)
        worst, flips = verify(torch_policy, numpy_policy, states)
        if worst > 1e-5:
            os.remove(path)
            raise SystemExit(f'{name}: numpy policy differs from torch by {worst:.2e}; not exported')
        # sample observations + expected probabilities for the web repo's own check
        check = states[:200]
        obs = np.stack([np.asarray(s[numpy_policy.obs_key], dtype=np.float64) for s in check])
        legal = np.zeros((len(check), 4), dtype=bool)
        for i, s in enumerate(check):
            legal[i, list(s['legal_actions'])] = True
        expected = np.stack([numpy_policy.probs(s) for s in check])
        np.savez_compressed(os.path.join(web, 'agents', f'{name}.check.npz'), obs=obs, legal=legal,
                            expected=expected)
        entries.append({'name': name, 'kind': kind, 'type': 'network', 'file': f'{name}.npz',
                        'encoding': encoding, 'description': ALGO_DESCRIPTIONS.get(kind, kind),
                        'source_checkpoint': _repo_path(ckpt)})
        print(f'agent {name:10s} {kind:8s} {encoding:6s} {output:14s} max |numpy - torch| = {worst:.1e}'
              + (f' ({flips} near-tied greedy choice(s))' if flips else ''))

    from eval_panel import NIT_PARAMS, LAG_PARAMS
    params = {'tag': {}, 'nit': NIT_PARAMS, 'lag': LAG_PARAMS}
    for bot in [b.strip() for b in a.bots.split(',') if b.strip()]:
        entry = {'name': bot, 'kind': bot, 'type': 'bot', 'description': BOT_DESCRIPTIONS[bot]}
        if bot in params:
            entry['params'] = params[bot]
        entries.append(entry)

    pool = ([n.strip() for n in a.random_pool.split(',')] if a.random_pool else
            [e['name'] for e in entries if e['type'] == 'network'] + [b for b in ('tag', 'nit', 'lag')
                                                                       if any(e['name'] == b for e in entries)])
    unknown = [n for n in pool if n not in {e['name'] for e in entries}]
    if unknown:
        raise SystemExit(f'--random_pool names unknown agents: {unknown}')
    # the agents folder belongs to this script: drop files of agents no longer exported
    keep = {f for e in entries if e['type'] == 'network' for f in (e['file'], e['file'][:-4] + '.check.npz')}
    for f in os.listdir(os.path.join(web, 'agents')):
        if f.endswith('.npz') and f not in keep:
            os.remove(os.path.join(web, 'agents', f))
            print(f'removed stale {f}')
    with open(os.path.join(web, 'agents', 'agents.json'), 'w', encoding='utf-8') as f:
        json.dump({'agents': entries, 'random_pool': pool, 'thesis_commit': commit,
                   'exported': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
    print(f'agents.json: {len(entries)} agents, random pool {pool}')


if __name__ == '__main__':
    main()
