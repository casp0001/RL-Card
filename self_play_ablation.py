''' Self-play opponent ablation: does a snapshot pool and/or a rule-based
warm-up give a more stable, less exploitable agent than plain latest-snapshot
self-play? (see self_play.py for the options)

Every variant trains the same configuration (a tuning study's best trial, or
--trial N) for the study's full budget with several seeds; each run is then
evaluated on deals that neither tuning nor the warm-up ever saw:
  panel  the evaluation panel on held-out deals (final_evaluation's seed)
  LBR    local best response exploitability (the measure the warm-up bot
         cannot bias: LBR plays none of the bots)
  play   500 duplicate deals against TAG recorded with behaviour_tracker:
         VPIP / PFR / aggression and fold decisions relative to pot odds

    python self_play_ablation.py run --study ppo_v1 --seeds 2 --workers 4 --threads 3
    python self_play_ablation.py report --study ppo_v1

Runs live in experiments/ablation/<study>/<variant>/seed_<S>/ (checkpoint,
training logs, result.json). `run` is resumable: finished runs are skipped,
unfinished ones restart from scratch.
'''

import os
import sys
import json
import math
import time
import shutil
import argparse
import subprocess

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ABLATION_ROOT = os.path.join(REPO_ROOT, 'experiments', 'ablation')
sys.path.insert(0, REPO_ROOT)

POOL = 10
VARIANTS = {
    # name: (description, overrides; 'W' = warm-up episodes, set from --warmup_frac)
    'latest': ('newest snapshot only (plain self-play)', {}),
    'pool': (f'pool of the last {POOL} snapshots, newest half of the time', {'opponent_pool': POOL}),
    'warmup_honest': ('warm-up vs HonestAgent, fading to 0', {'warmup_episodes': 'W', 'warmup_bot': 'honest'}),
    'warmup_mix': ('warm-up vs randomised equity bot, fading to 0',
                   {'warmup_episodes': 'W', 'warmup_bot': 'equity_mix'}),
    'warmup_mix_pool': ('randomised equity warm-up + snapshot pool',
                        {'warmup_episodes': 'W', 'warmup_bot': 'equity_mix', 'opponent_pool': POOL}),
}
SUPPORTED = ('dqn', 'ppo', 'nfsp')
SELF_PLAY_DEFAULTS = {'opponent_pool': 1, 'pool_latest_frac': 0.5, 'warmup_episodes': 0, 'warmup_bot': 'honest',
                      'warmup_fade': 1, 'warmup_start_frac': 1.0}


def study_root(study):
    return os.path.join(ABLATION_ROOT, study)


def run_dir(study, variant, seed):
    return os.path.join(study_root(study), variant, f'seed_{seed}')


def best_params(study, trial=None):
    import tune
    import optuna
    from optuna.trial import TrialState
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    s = tune.load_study(study)
    complete = [t for t in s.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
    if trial is not None:
        chosen = [t for t in s.get_trials(deepcopy=False) if t.number == trial]
        if not chosen:
            raise SystemExit(f'study {study} has no trial {trial}')
        return chosen[0].number, chosen[0].params
    if not complete:
        raise SystemExit(f'study {study} has no completed trial yet')
    best = max(complete, key=lambda t: t.value)
    return best.number, best.params


# --------------------------------------------------------------------------- #
# One run (called in a subprocess by `run`)
# --------------------------------------------------------------------------- #

def cmd_train(a):
    import tune
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = tune.load_config(a.study)
    trial, params = best_params(a.study, a.trial)
    out = run_dir(a.study, a.variant, a.seed)
    module, args = tune.build_args(cfg, params, out, a.seed, save_model=True)
    _, overrides = VARIANTS[a.variant]
    warmup = int(round(a.warmup_frac * cfg['budget']))
    # a study may fix self-play options itself (tune.py --set, e.g. ppo_v2's
    # warm-up): every variant starts from plain self-play, then adds its own
    for key, value in SELF_PLAY_DEFAULTS.items():
        if hasattr(args, key):
            setattr(args, key, value)
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise SystemExit(f'{module.__name__} has no option {key}')
        setattr(args, key, warmup if value == 'W' else value)
    t0 = time.time()
    agent = module.train(args, None)
    train_seconds = time.time() - t0
    del agent

    from heldout_eval import evaluate
    held = evaluate(cfg['algo'], out, a.panel_deals, a.lbr_deals, a.play_deals, name=a.variant)
    result = {'study': a.study, 'trial': trial, 'params': params, 'variant': a.variant, 'seed': a.seed,
              'overrides': {k: (warmup if v == 'W' else v) for k, v in overrides.items()},
              'self_play': {k: getattr(args, k) for k in SELF_PLAY_DEFAULTS if hasattr(args, k)},
              'train_seconds': train_seconds, **held}
    with open(os.path.join(out, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1)
    print(f'done {a.variant} seed {a.seed}: panel min {held["panel"]["min"]:+.0f}, '
          f'LBR {held["lbr"]["mean"]:+.0f}', flush=True)


# --------------------------------------------------------------------------- #
# All runs
# --------------------------------------------------------------------------- #

def cmd_run(a):
    import tune
    cfg = tune.load_config(a.study)
    if cfg['algo'] not in SUPPORTED:
        raise SystemExit(f'{cfg["algo"]} does not play opponents; the ablation is for {SUPPORTED}')
    trial, params = best_params(a.study, a.trial)
    variants = [v.strip() for v in a.variants.split(',')] if a.variants else list(VARIANTS)
    seeds = [cfg['seed'] + 2000 + i for i in range(a.seeds)]
    print(f'{a.study}: trial {trial} {json.dumps(params)}\n'
          f'{len(variants)} variants x {len(seeds)} seeds, {a.workers} at a time', flush=True)
    queue = []
    for seed in seeds:                        # seed-major: every variant gets its first seed early
        for v in variants:
            out = run_dir(a.study, v, seed)
            if os.path.isfile(os.path.join(out, 'result.json')):
                print(f'  {v} seed {seed}: done, skipping')
                continue
            queue.append((v, seed, out))
    running = []
    while queue or running:
        while queue and len(running) < a.workers:
            v, seed, out = queue.pop(0)
            shutil.rmtree(out, ignore_errors=True)       # a half-finished run restarts from scratch
            os.makedirs(out)
            cmd = [sys.executable, '-u', os.path.abspath(__file__), 'train', '--study', a.study,
                   '--variant', v, '--seed', str(seed), '--threads', str(a.threads),
                   '--warmup_frac', str(a.warmup_frac), '--panel_deals', str(a.panel_deals),
                   '--lbr_deals', str(a.lbr_deals), '--play_deals', str(a.play_deals)]
            if a.trial is not None:
                cmd += ['--trial', str(a.trial)]
            log = open(os.path.join(out, 'run.log'), 'w', encoding='utf-8')
            running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT), v, seed,
                            log, time.time()))
            print(f'  started {v} seed {seed}', flush=True)
        time.sleep(10)
        for item in list(running):
            proc, v, seed, log, start = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                status = 'finished' if proc.returncode == 0 else f'FAILED (exit {proc.returncode}, see run.log)'
                print(f'  {v} seed {seed}: {status} after {(time.time() - start) / 60:.0f} min', flush=True)
    report(a)


def _agg(values):
    values = [v for v in values if v is not None]
    if not values:
        return ''
    m = sum(values) / len(values)
    if len(values) == 1:
        return f'{m:+.0f}' if abs(m) >= 2 else f'{m:.1%}'
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))
    return f'{m:+.0f} (sd {sd:.0f})' if abs(m) >= 2 else f'{m:.1%} (sd {sd:.1%})'


def report(a):
    root = study_root(a.study)
    rows, header = [], None
    for v, (desc, _) in VARIANTS.items():
        results = []
        vdir = os.path.join(root, v)
        if os.path.isdir(vdir):
            for s in sorted(os.listdir(vdir)):
                p = os.path.join(vdir, s, 'result.json')
                if os.path.isfile(p):
                    with open(p, encoding='utf-8') as f:
                        results.append(json.load(f))
        if not results:
            continue
        members = list(results[0]['panel']['members'])
        header = (['variant', 'seeds', 'panel min', 'LBR (lower = better)'] + [f'vs {m}' for m in members]
                  + ['fold facing bet', 'fold when behind', 'fold when well ahead', 'VPIP', 'PFR', 'train min'])

        def play(r, key):
            v = r['play_vs_tag'].get(key)
            return v.get('value') if isinstance(v, dict) else None
        rows.append([v, len(results), _agg([r['panel']['min'] for r in results]),
                     _agg([r['lbr']['mean'] for r in results])]
                    + [_agg([r['panel']['members'][m]['mean'] for r in results]) for m in members]
                    + [_agg([play(r, k) for r in results]) for k in
                       ('fold facing a bet', 'fold when equity < pot odds', 'fold when equity >> pot odds',
                        'VPIP', 'PFR')]
                    + [f'{sum(r["train_seconds"] for r in results) / len(results) / 60:.0f}'])
    if not rows:
        print('no finished runs yet')
        return
    lines = [f'# Self-play ablation: {a.study}', '',
             'Held-out panel deals and LBR (mbb/hand); fold rates from 500 duplicate deals vs TAG '
             '("behind": equity below the pot odds; "well ahead": 15 points above). '
             'Mean over seeds (sd = spread over seeds).', '',
             '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    lines += ['', 'Variants:'] + [f'- {v}: {d}' for v, (d, _) in VARIANTS.items()]
    path = os.path.join(root, 'summary.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nwritten to {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    t = sub.add_parser('train')
    for q in (r, t):
        q.add_argument('--study', required=True)
        q.add_argument('--trial', type=int, default=None, help='trial to use (default: the best completed one)')
        q.add_argument('--threads', type=int, default=3)
        q.add_argument('--warmup_frac', type=float, default=0.2,
                       help='warm-up length as a fraction of the budget (the bot share fades to 0 over it)')
        q.add_argument('--panel_deals', type=int, default=2000)
        q.add_argument('--lbr_deals', type=int, default=1000)
        q.add_argument('--play_deals', type=int, default=500)
    r.add_argument('--seeds', type=int, default=2)
    r.add_argument('--workers', type=int, default=4)
    r.add_argument('--variants', default=None, help='comma-separated subset of: ' + ', '.join(VARIANTS))
    t.add_argument('--variant', required=True, choices=list(VARIANTS))
    t.add_argument('--seed', type=int, required=True)
    g = sub.add_parser('report')
    g.add_argument('--study', required=True)
    a = p.parse_args()
    {'run': cmd_run, 'train': cmd_train, 'report': report}[a.command](a)


if __name__ == '__main__':
    main()
