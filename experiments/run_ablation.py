''' Ablations of two findings of the code review, each on one tuned winner
(the finalist configuration and seeds, everything else identical). They
decide whether any algorithm has to be retrained, so they run before the
final evaluation (experiments/run_ablations.py runs both).

  ppo_free_fold   RLCard always leaves fold legal, even when a check costs
                  nothing; the agents keep some probability on that dominated
                  fold (PPO the most). ppo_v2 winner, three arms:
                    A  original    the finalist runs as trained
                    B  corrected   A with its fold probability moved to check where
                                   checking is legal, at evaluation only (no retraining)
                    C  retrained   trained again with fold illegal when checking is legal
                  B - A: what the wasted fold probability costs; C - B: whether the
                  rule also changes what the agent learns (C ~ B: an evaluation-time
                  correction suffices for every algorithm).
  nfsp_batchnorm  rlcard's networks start with BatchNorm, which normalises with
                  minibatch statistics in training and with running averages in
                  play, so NFSP (and DQN) fit one function and play another.
                  nfsp_v3 winner, two arms:
                    A  original    the finalist runs (BatchNorm in both networks)
                    C  retrained   trained again without BatchNorm in either network
                  C - A: what BatchNorm costs NFSP (DQN shares the network code; a
                  large effect would make retraining DQN worth its ~8 h per run).

Two follow-ups on whether Deep CFR and NFSP were held back by their
configuration (both far below the published scale, see the thesis notes):

  deepcfr_sgd     Brown et al. (2019): fewer SGD steps per network fit do not slow
                  convergence but raise the asymptotic exploitability; the study
                  fixed num_step = 1000 (the paper: 32,000 for limit hold'em) and the
                  extended run plateaued from ~iteration 320. deepcfr_v1 winner,
                  200 iterations, two arms:
                    A  original    the finalist runs (1000 SGD steps per fit)
                    C  retrained   4000 SGD steps per fit, everything else identical
  nfsp_literature the nfsp_v3 winner sits at the lower bound of four parameters
                  (eta, RL learning rate, epsilon, train_every) and its reservoir is far
                  below the published sizes. 8M episodes (the extended budget), two arms:
                    A  tuned       the extended nfsp_v3 winner (nfsp_ext)
                    C  literature  the same run with the published values where the
                                   winner sat at a bound or far from them: eta 0.1,
                                   epsilon start 0.08 (Heinrich & Silver 2016), RL
                                   learning rate 1e-4 (Adam; Lanctot et al. 2017's best,
                                   inside nfsp_v3's divergence-free range), reservoir
                                   1.5M (the RAM cap; papers 2M-30M), replay 200k;
                                   network, SL learning rate and train_every as tuned
Both also measure, and play, the extended PPO and Deep CFR runs (ppo_ext,
deepcfr_ext): the h2h winner and the least exploitable agent of the final
evaluation, on the same deals.

Every arm gets the same measurement (the held-out protocol of heldout_eval.py):
held-out panel (FINAL_PANEL_SEED deals), LBR, a match against tag for the
style statistics (incl. P(fold | check free)), and duplicate head-to-head
matches between the arms and against the reference agents (the Deep CFR
winner, or for the follow-ups ppo_ext and deepcfr_ext).

    python experiments/run_ablation.py ppo_free_fold              (train the missing runs, then evaluate)
    python experiments/run_ablation.py nfsp_batchnorm --skip_train  (evaluate only)
Resumable: finished training runs and finished measurements are kept.
Output: experiments/tune/<study>/<variant>_<budget>/trial_<n>/seed_*/,
        experiments/ablation_<name>.md / .json (+ .csv / .tex per table)
'''
import os
import sys
import json
import time
import argparse
import statistics
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

DEEPCFR = 'deepcfr_v1'
# reference agents: (label, study, run folder under the study, kind); every arm plays every reference
DEEPCFR_REF = [('deepcfr', DEEPCFR, 'finalists', 'deepcfr')]
EXTENDED_REFS = [('ppo_ext', 'ppo_v2', 'extended_4000000', 'ppo'),
                 ('deepcfr_ext', DEEPCFR, 'extended_500', 'deepcfr')]
# optional spec keys: 'params' (overrides of the winner's tuned values), 'budget' (training budget,
# default the study's), 'baseline' (run folder of arm A, default 'finalists'), 'refs' (default
# DEEPCFR_REF), 'measure_refs' (also measure the references; default False)
ABLATIONS = {
    'ppo_free_fold': {
        'study': 'ppo_v2', 'kind': 'ppo', 'variant': 'no_free_fold', 'fixed': {'no_free_fold': True},
        'title': 'Free-fold ablation, PPO',
        'arms': [('A', 'original (free fold allowed)'), ('B', 'A, fold moved to check at evaluation'),
                 ('C', 'retrained without free fold')],
    },
    'nfsp_batchnorm': {
        'study': 'nfsp_v3', 'kind': 'nfsp', 'variant': 'no_batchnorm', 'fixed': {'batchnorm': False},
        'title': 'BatchNorm ablation, NFSP',
        'arms': [('A', 'original (BatchNorm on the input of both networks)'),
                 ('C', 'retrained without BatchNorm')],
    },
    'deepcfr_sgd': {
        'study': DEEPCFR, 'kind': 'deepcfr', 'variant': 'sgd4000', 'fixed': {'num_step': 4000},
        'title': 'SGD-steps ablation, Deep CFR',
        'arms': [('A', 'original (1000 SGD steps per network fit)'),
                 ('C', 'retrained with 4000 SGD steps per fit')],
        'refs': EXTENDED_REFS, 'measure_refs': True,
    },
    'nfsp_literature': {
        'study': 'nfsp_v3', 'kind': 'nfsp', 'variant': 'literature', 'budget': 8_000_000,
        'baseline': 'extended_8000000', 'fixed': {},
        'params': {'anticipatory_param': 0.1, 'q_epsilon_start': 0.08, 'rl_learning_rate': 1e-4,
                   'reservoir_buffer_capacity': 1_500_000, 'q_replay_memory_size': 200_000},
        'title': 'Literature-configuration check, NFSP (8M episodes)',
        'arms': [('A', 'tuned (nfsp_v3 winner, extended)'),
                 ('C', 'literature values (eta 0.1, eps 0.08, RL lr 1e-4, reservoir 1.5M, replay 200k)')],
        'refs': EXTENDED_REFS, 'measure_refs': True,
    },
}


def winner(study):
    with open(os.path.join('experiments', 'tune', study, 'finalists', 'finalists.json'), encoding='utf-8') as f:
        return json.load(f)['winner']


def run_dir(study, folder, seed):
    return os.path.join('experiments', 'tune', study, folder, f'trial_{winner(study)["trial"]}', f'seed_{seed}')


def finalist_dir(study, seed):
    return run_dir(study, 'finalists', seed)


def baseline_dir(a, seed):
    ''' Arm A: the finalist run, or (spec 'baseline') e.g. the extended run of the same winner. '''
    return run_dir(a.spec['study'], a.spec.get('baseline', 'finalists'), seed)


def refs(a):
    return a.spec.get('refs', DEEPCFR_REF)


def ref_seeds(label, a):
    return winner(next(r for r in refs(a) if r[0] == label)[1])['seeds']


def default_budget(spec):
    import tune
    return spec.get('budget') or tune.load_config(spec['study'])['budget']


def retrained_dir(a, seed):
    return os.path.join(a.root, f'trial_{winner(a.spec["study"])["trial"]}', f'seed_{seed}')


# --------------------------------------------------------------------------- #
# Training (arm C)
# --------------------------------------------------------------------------- #

def cmd_train_one(a):
    ''' One retrained run: the winner's configuration with the ablation's change. '''
    import torch
    import tune
    import tune_spaces
    torch.set_num_threads(a.threads)
    study = a.spec['study']
    cfg = tune.load_config(study)
    interval = cfg['budget'] // cfg['evals']
    budget = a.budget or default_budget(a.spec)
    cfg = dict(cfg, budget=budget, evals=max(1, budget // interval), fixed=dict(cfg['fixed'], **a.spec['fixed']))
    w = winner(study)
    over = a.spec.get('params', {})
    bad = [k for k in over if k not in cfg['space'] or not cfg['space'][k].get('tune', True)]
    assert not bad, f'params {bad} are not tuned parameters of {study} (use fixed)'
    params = dict(w['params'], **over)
    log_dir = retrained_dir(a, a.seed)
    module, args = tune.build_args(cfg, params, log_dir, a.seed, save_model=True)
    # build_args applies the space's fixed values after the study's fixed settings (e.g. Deep CFR's
    # num_step), so the ablation's own changes are set last
    for k, v in a.spec['fixed'].items():
        setattr(args, k, v)
    for name, value in over.items():
        for k, v in tune_spaces.to_args(name, cfg['space'][name], value).items():
            assert getattr(args, k) == v, (k, getattr(args, k), v)
    for k, v in a.spec['fixed'].items():
        assert getattr(args, k) == v, (k, getattr(args, k))
    print(f'{a.ablation} seed {a.seed}: budget {budget}, changes {dict(a.spec["fixed"], **over)}', flush=True)
    history = []

    def track(episode, results):
        history.append(results['min'])
        return False
    start = time.time()
    module.train(args, track)
    with open(os.path.join(log_dir, 'trained.json'), 'w', encoding='utf-8') as f:
        json.dump({'ablation': a.ablation, 'study': study, 'trial': w['trial'], 'seed': a.seed, 'budget': budget,
                   'change': dict(a.spec['fixed'], **over), 'params': params, 'history': history,
                   'hours': (time.time() - start) / 3600}, f, indent=1)


def train_all(a):
    seeds = winner(a.spec['study'])['seeds']
    todo = [s for s in seeds if not os.path.isfile(os.path.join(retrained_dir(a, s), 'trained.json'))]
    print(f'{a.ablation}: {len(seeds) - len(todo)} of {len(seeds)} retrained runs done; training {todo}', flush=True)
    procs = []
    for s in todo:
        d = retrained_dir(a, s)
        os.makedirs(d, exist_ok=True)
        log = open(os.path.join(d, 'run.log'), 'w', encoding='utf-8')
        cmd = ([sys.executable, '-u', os.path.abspath(__file__), a.ablation, 'train_one', '--seed', str(s),
                '--root', a.root, '--threads', str(a.threads)] + (['--budget', str(a.budget)] if a.budget else []))
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO), s, log))
    failed = []
    for p, s, log in procs:
        code = p.wait()
        log.close()
        print(f'{time.strftime("%Y-%m-%d %H:%M")}  {a.ablation} seed {s}: '
              + ('done' if code == 0 else f'FAILED (exit {code}, see {retrained_dir(a, s)}/run.log)'), flush=True)
        failed += [s] if code else []
    if failed:
        raise SystemExit(f'{a.ablation}: training failed for seeds {failed}')


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #

def agents(a):
    ''' name -> (arm, seed, kind, checkpoint dir, evaluation-time free-fold correction) '''
    study, kind = a.spec['study'], a.spec['kind']
    arms = [arm for arm, _ in a.spec['arms']]
    out = {}
    for s in winner(study)['seeds']:
        out[f'A_s{s}'] = ('A', s, kind, baseline_dir(a, s), False)
        if 'B' in arms:
            out[f'B_s{s}'] = ('B', s, kind, baseline_dir(a, s), True)
        out[f'C_s{s}'] = ('C', s, kind, retrained_dir(a, s), False)
    for label, ref_study, folder, ref_kind in refs(a):
        for s in winner(ref_study)['seeds']:
            out[f'{label}_s{s}'] = (label, s, ref_kind, run_dir(ref_study, folder, s), False)
    return out


def load(spec):
    from agent_loading import load_policy, NoFreeFoldPolicy
    arm, seed, kind, d, corrected = spec
    p = load_policy(kind, d)
    return NoFreeFoldPolicy(p) if corrected else p


def cmd_measure_one(a):
    ''' Panel, LBR and style statistics of one agent (in its own process). '''
    import torch
    import behaviour_tracker as bt
    from agent_loading import load_policy
    from eval_panel import Panel, DEFAULT_MEMBERS
    from exploitability import LBR
    from final_evaluation import FINAL_PANEL_SEED
    from heldout_eval import fold_stats
    torch.set_num_threads(a.threads)
    policy = load(agents(a)[a.name])
    panel = Panel(DEFAULT_MEMBERS, num_deals=a.panel_deals, eval_seed=FINAL_PANEL_SEED).evaluate(policy)
    lbr = LBR(policy, samples=500).evaluate(a.lbr_deals)
    hands = bt.play_match(policy, load_policy('tag'), a.play_deals, names=(a.name, 'tag'))
    stats = bt.compute_stats(hands)
    stats.update(fold_stats(hands))
    free = [d['probs']['fold'] if d.get('probs') else float(d['action'] == 'fold')
            for h in hands for d in h['decisions'] if d['role'] == 'agent' and 'check' in d['legal']]
    out = {'panel': {'min': panel['min'], 'mean': panel['mean'],
                     'members': {k: v['mean'] for k, v in panel['members'].items()}},
           'lbr': {'mean': lbr['mean'], 'se': lbr['se']},
           'free_fold': {'mean': statistics.fmean(free) if free else None, 'n': len(free)},
           'style': {k: stats[k][0] for k in ('VPIP', 'PFR', 'fold facing a bet', 'aggression factor (post-flop)')}}
    path = os.path.join(a.work, f'{a.name}.json')
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    os.replace(path + '.tmp', path)


def measure_all(a):
    os.makedirs(a.work, exist_ok=True)
    measured = {arm for arm, _ in a.spec['arms']} | ({r[0] for r in refs(a)} if a.spec.get('measure_refs') else set())
    names = [n for n, spec in agents(a).items() if spec[0] in measured]
    todo = [n for n in names if not os.path.isfile(os.path.join(a.work, f'{n}.json'))]
    print(f'{a.ablation}: measurements {len(names) - len(todo)} of {len(names)} done', flush=True)
    running = []
    for n in todo:
        while len(running) >= a.processes:
            running = [p for p in running if p.poll() is None]
            time.sleep(2)
        log = open(os.path.join(a.work, f'{n}.log'), 'w', encoding='utf-8')
        cmd = [sys.executable, '-u', os.path.abspath(__file__), a.ablation, 'measure_one', '--name', n,
               '--root', a.root, '--work', a.work, '--threads', str(a.threads), '--panel_deals', str(a.panel_deals),
               '--lbr_deals', str(a.lbr_deals), '--play_deals', str(a.play_deals)]
        running.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO))
    for p in running:
        p.wait()
    missing = [n for n in names if not os.path.isfile(os.path.join(a.work, f'{n}.json'))]
    if missing:
        raise SystemExit(f'{a.ablation}: measurement failed for {missing} (see {a.work}/<name>.log)')


def _pairs(a):
    seeds = winner(a.spec['study'])['seeds']
    arms = [arm for arm, _ in a.spec['arms']]
    between = [(x, y) for x, y in (('C', 'A'), ('C', 'B'), ('B', 'A')) if x in arms and y in arms]
    return between + [(x, r[0]) for r in refs(a) for x in arms], seeds


def h2h_all(a):
    ''' Duplicate head-to-head: the arms against each other and against Deep CFR. '''
    from eval_panel import duplicate_match
    path = os.path.join(a.work, 'h2h.json')
    done = {}
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            done = json.load(f)
    specs = agents(a)
    groups, seeds = _pairs(a)
    ref_labels = {r[0] for r in refs(a)}
    pairs = [(f'{x}_s{s}', f'{y}_s{t}') for x, y in groups for s in seeds
             for t in (ref_seeds(y, a) if y in ref_labels else seeds)]
    todo = [p for p in pairs if f'{p[0]}:{p[1]}' not in done]
    print(f'{a.ablation}: head-to-head {len(pairs) - len(todo)} of {len(pairs)} matches done', flush=True)
    policies = {}
    for i, (x, y) in enumerate(todo, 1):
        for n in (x, y):
            if n not in policies:
                policies[n] = load(specs[n])
        r = duplicate_match(policies[x], policies[y], a.h2h_deals)
        done[f'{x}:{y}'] = {'mean': r['mean'], 'se': r['se']}
        with open(path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(done, f, indent=1)
        os.replace(path + '.tmp', path)
        print(f'  [{i}/{len(todo)}] {x} vs {y}: {r["mean"]:+.0f} +/- {r["se"]:.0f}', flush=True)
    return done


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _agg(xs, digits=0, pct=False):
    xs = [x for x in xs if x is not None]
    if not xs:
        return ''
    k = 100 if pct else 1
    s = f'{k * statistics.fmean(xs):.{digits}f}%' if pct else f'{statistics.fmean(xs):+.{digits}f}'
    return s + (f' (sd {k * statistics.stdev(xs):.{digits}f})' if len(xs) > 1 else '')


def report(a, h2h):
    from final_evaluation import _write_table, _h2h_algo_se
    import numpy as np
    study = a.spec['study']
    seeds = winner(study)['seeds']
    m = {}
    for n in agents(a):
        p = os.path.join(a.work, f'{n}.json')
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as f:
                m[n] = json.load(f)
    ref_labels = [r[0] for r in refs(a)]
    rows_spec = list(a.spec['arms']) + ([(r, f'reference ({r})') for r in ref_labels]
                                        if a.spec.get('measure_refs') else [])
    members = list(next(iter(m.values()))['panel']['members'])
    header = (['arm', 'P(fold | check free)', 'panel min', 'panel mean'] + [f'vs {x}' for x in members]
              + ['LBR', 'VPIP', 'PFR', 'fold facing a bet', 'AF'])
    rows = []
    for arm, label in rows_spec:
        arm_seeds = ref_seeds(arm, a) if arm in ref_labels else seeds
        ms = [m[f'{arm}_s{s}'] for s in arm_seeds if f'{arm}_s{s}' in m]
        if not ms:
            continue
        rows.append([label if arm in ref_labels else f'{arm}: {label}',
                     _agg([x['free_fold']['mean'] for x in ms], 1, pct=True),
                     _agg([x['panel']['min'] for x in ms]), _agg([x['panel']['mean'] for x in ms])]
                    + [_agg([x['panel']['members'][k] for x in ms]) for k in members]
                    + [_agg([x['lbr']['mean'] for x in ms])]
                    + [_agg([x['style'][k] for x in ms], 0, pct=True) for k in ('VPIP', 'PFR', 'fold facing a bet')]
                    + [_agg([x['style']['aggression factor (post-flop)'] for x in ms], 2).lstrip('+')])
    md = [_write_table(a.out, f'{a.spec["title"]} ({study} trial {winner(study)["trial"]}), mean (sd) over seeds {seeds}',
                       header, rows,
                       f'Held-out panel ({a.panel_deals} duplicate deals per bot, FINAL_PANEL_SEED), LBR '
                       f'({a.lbr_deals} deals, lower = less exploitable), style from {a.play_deals} deals against tag. '
                       'mbb/hand.')]

    # paired differences per seed (same seed = same initialisation and data stream up to the change)
    groups, _ = _pairs(a)
    rows = []
    for x, y in [g for g in groups if g[1] not in ref_labels]:
        def diffs(f):
            return [f(m[f'{x}_s{s}']) - f(m[f'{y}_s{s}']) for s in seeds if f'{x}_s{s}' in m and f'{y}_s{s}' in m]
        rows.append([f'{x} - {y}', _agg(diffs(lambda r: r['panel']['min'])),
                     _agg(diffs(lambda r: r['panel']['mean'])), _agg(diffs(lambda r: r['lbr']['mean']))])
    md.append(_write_table(a.out + '_diff', 'Paired differences per seed (mean (sd) over seeds)',
                           ['difference', 'panel min', 'panel mean', 'LBR (negative = less exploitable)'], rows))

    rows = []
    for x, y in groups:
        ys = ref_seeds(y, a) if y in ref_labels else seeds
        grid = np.full((len(seeds), len(ys)), np.nan)
        var = []
        for i, s in enumerate(seeds):
            for j, t in enumerate(ys):
                r = h2h.get(f'{x}_s{s}:{y}_s{t}')
                if r:
                    grid[i, j] = r['mean']
                    var.append(r['se'] ** 2)
        if var:
            rows.append([f'{x} vs {y}', f'{np.nanmean(grid):+.0f} ± {_h2h_algo_se(grid, var):.0f}'])
    md.append(_write_table(a.out + '_h2h',
                           f'Head-to-head ({a.h2h_deals} duplicate deals per pair, mean over seed pairs)',
                           ['match', 'row wins (mbb/hand)'], rows,
                           '± counts the training seeds of both sides as the random units plus the card noise.'))

    # training-time score (what tuning would have seen: mean of the last 3 panel evaluations)
    a_scores, a_hours = [], []
    if a.spec.get('baseline', 'finalists') == 'finalists':
        with open(os.path.join('experiments', 'tune', study, 'finalists', 'finalists.json'), encoding='utf-8') as f:
            runs = {r['seed']: r for r in json.load(f)['runs'] if r['trial'] == winner(study)['trial']}
        a_scores = [runs[s].get('training_score') for s in seeds if s in runs]
    else:   # an extended run: extended.json holds its panel history and wall time
        for s in seeds:
            p = os.path.join(baseline_dir(a, s), 'extended.json')
            if os.path.isfile(p):
                with open(p, encoding='utf-8') as f:
                    e = json.load(f)
                a_scores.append(statistics.fmean(e['history'][-3:]))
                a_hours.append(e.get('hours'))
    c_scores, hours = [], []
    for s in seeds:
        p = os.path.join(retrained_dir(a, s), 'trained.json')
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as f:
                t = json.load(f)
            c_scores.append(statistics.fmean(t['history'][-3:]))
            hours.append(t['hours'])
    md.append(_write_table(a.out + '_training', 'Training-time panel score (mean of the last 3 evaluations, tuning deals)',
                           ['arm', 'score', 'training hours'],
                           [['A: ' + a.spec['arms'][0][1], _agg(a_scores), _agg(a_hours, 1).lstrip('+')],
                            ['C: retrained', _agg(c_scores), _agg(hours, 1).lstrip('+')]]))
    with open(a.out + '.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    with open(a.out + '.json', 'w', encoding='utf-8') as f:
        json.dump({'ablation': a.ablation, 'measurements': m, 'h2h': h2h,
                   'settings': {k: v for k, v in vars(a).items() if k != 'spec'}}, f, indent=1, default=str)
    print('\n'.join(md))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('ablation', choices=sorted(ABLATIONS))
    p.add_argument('command', nargs='?', default='all', choices=['all', 'train_one', 'measure_one'])
    p.add_argument('--root', default=None, help='folder of the retrained runs (default: under the study)')
    p.add_argument('--work', default=None, help='per-agent measurements and head-to-head results')
    p.add_argument('--out', default=None, help='path prefix of the result tables (.md / .json / .csv / .tex)')
    p.add_argument('--budget', type=int, default=None, help="training budget (default: the study's)")
    p.add_argument('--seed', type=int)
    p.add_argument('--name')
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--processes', type=int, default=3, help='measurements run at once')
    p.add_argument('--panel_deals', type=int, default=2000)
    p.add_argument('--lbr_deals', type=int, default=1000)
    p.add_argument('--play_deals', type=int, default=500)
    p.add_argument('--h2h_deals', type=int, default=2000)
    p.add_argument('--skip_train', action='store_true')
    a = p.parse_args()
    a.spec = ABLATIONS[a.ablation]
    if a.root is None:
        budget = a.budget or default_budget(a.spec)
        a.root = os.path.join('experiments', 'tune', a.spec['study'], f'{a.spec["variant"]}_{budget}')
    a.work = a.work or os.path.join('experiments', f'ablation_{a.ablation}_work')
    a.out = a.out or os.path.join('experiments', f'ablation_{a.ablation}')
    if a.command == 'train_one':
        return cmd_train_one(a)
    if a.command == 'measure_one':
        return cmd_measure_one(a)
    if not a.skip_train:
        train_all(a)
    measure_all(a)
    report(a, h2h_all(a))


if __name__ == '__main__':
    main()
