''' Extended training of a study's winner: the finalists' configuration and
seeds, only with a larger budget (the evaluation interval stays the study's,
so the learning curve continues the finalists' one). Each run is trained in
its own process, then evaluated with heldout_eval.py; the table compares the
extended runs with the study's finalists of the same trial.

    python experiments/run_extended.py --run deepcfr_v1:500 --run nfsp_v3:8000000 [--processes 4]

Runs start in the order given (seeds inside a study), --processes at a time.
Resumable: a finished run (heldout.json present) is skipped, an unfinished
one restarts from scratch.
Output: experiments/tune/<study>/extended_<budget>/trial_<n>/seed_<s>/
'''
import os
import sys
import json
import time
import shutil
import argparse
import statistics
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)


def winner(study):
    with open(os.path.join('experiments', 'tune', study, 'finalists', 'finalists.json'), encoding='utf-8') as f:
        return json.load(f)['winner']


def run_dir(study, budget, trial, seed):
    return os.path.join('experiments', 'tune', study, f'extended_{budget}', f'trial_{trial}', f'seed_{seed}')


def cmd_one(a):
    ''' Train one extended run (called in a subprocess by cmd_all). '''
    import torch
    import tune
    torch.set_num_threads(a.threads)
    cfg = tune.load_config(a.study)
    w = winner(a.study)
    interval = cfg['budget'] // cfg['evals']
    cfg = dict(cfg, budget=a.budget, evals=a.budget // interval)
    log_dir = run_dir(a.study, a.budget, w['trial'], a.seed)
    module, args = tune.build_args(cfg, w['params'], log_dir, a.seed, save_model=True)
    history = []

    def track(episode, results):
        history.append(results['min'])
        return False

    start = time.time()
    module.train(args, track)
    with open(os.path.join(log_dir, 'extended.json'), 'w', encoding='utf-8') as f:
        json.dump({'study': a.study, 'trial': w['trial'], 'seed': a.seed, 'budget': a.budget,
                   'evaluate_every': interval, 'params': w['params'], 'history': history,
                   'hours': (time.time() - start) / 3600}, f, indent=1)


def _heldout(d):
    p = os.path.join(d, 'heldout.json')
    if os.path.isfile(p):
        with open(p, encoding='utf-8') as f:
            return json.load(f)


def _fmt(xs):
    if not xs:
        return '-'
    return f'{statistics.fmean(xs):+.0f}' + (f' (sd {statistics.stdev(xs):.0f})' if len(xs) > 1 else '')


def table(jobs):
    lines = ['| study | trial | budget | seeds | held-out panel min | LBR (lower = better) | training hours |',
             '|---|---|---|---|---|---|---|']
    for study, budget in jobs:
        import tune
        cfg = tune.load_config(study)
        w = winner(study)
        for b, dirs in ((cfg['budget'], [os.path.join('experiments', 'tune', study, 'finalists', f'trial_{w["trial"]}',
                                                       f'seed_{s}') for s in w['seeds']]),
                        (budget, [run_dir(study, budget, w['trial'], s) for s in w['seeds']])):
            h = [x for x in map(_heldout, dirs) if x]
            hours = []
            for d in dirs:
                p = os.path.join(d, 'extended.json')
                if os.path.isfile(p):
                    with open(p, encoding='utf-8') as f:
                        hours.append(json.load(f)['hours'])
            lines.append(f'| {study} | {w["trial"]} | {b} | {len(h)} | {_fmt([x["panel"]["min"] for x in h])} | '
                         f'{_fmt([x["lbr"]["mean"] for x in h])} | {f"{statistics.fmean(hours):.1f}" if hours else "-"} |')
    return '\n'.join(lines)


def cmd_all(a):
    import tune
    jobs = [(s, int(b)) for s, b in (r.split(':') for r in a.run)]
    queue = []
    for study, budget in jobs:
        w = winner(study)
        for seed in w['seeds']:
            d = run_dir(study, budget, w['trial'], seed)
            if os.path.isfile(os.path.join(d, 'heldout.json')):
                print(f'{d}: done, skipping', flush=True)
                continue
            queue.append((study, budget, seed, d, tune.load_config(study)['algo']))
    print(f'{len(queue)} extended run(s) to do', flush=True)
    running = []
    while queue or running:
        while queue and len(running) < a.processes:
            study, budget, seed, d, kind = queue.pop(0)
            if not os.path.isfile(os.path.join(d, 'extended.json')):
                shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
            log = open(os.path.join(d, 'run.log'), 'a', encoding='utf-8')
            # train (unless a previous start finished training), then the held-out evaluation
            train = [sys.executable, '-u', 'experiments/run_extended.py', 'one', '--study', study,
                     '--budget', str(budget), '--seed', str(seed), '--threads', str(a.threads)]
            heldout = [sys.executable, '-u', 'heldout_eval.py', 'one', '--kind', kind, '--dir', d,
                       '--threads', str(a.threads)]
            steps = [heldout] if os.path.isfile(os.path.join(d, 'extended.json')) else [train, heldout]
            proc = subprocess.Popen(steps[0], stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
            running.append([proc, steps[1:], d, log, time.time()])
            print(f'{time.strftime("%Y-%m-%d %H:%M")}  started {d}', flush=True)
        time.sleep(30)
        for item in list(running):
            proc, rest, d, log, t0 = item
            if proc.poll() is None:
                continue
            if proc.returncode == 0 and rest:
                item[0] = subprocess.Popen(rest[0], stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
                item[1] = rest[1:]
                continue
            log.close()
            running.remove(item)
            status = 'done' if proc.returncode == 0 else f'FAILED (exit {proc.returncode}, see run.log)'
            print(f'{time.strftime("%Y-%m-%d %H:%M")}  {d}: {status} after {(time.time() - t0) / 3600:.1f} h',
                  flush=True)
    text = table(jobs)
    out = os.path.join('experiments', 'extended_' + '_vs_'.join(s for s, _ in jobs) + '.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text + f'\n\nwritten to {out}', flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command')
    o = sub.add_parser('one')
    o.add_argument('--study', required=True)
    o.add_argument('--budget', type=int, required=True)
    o.add_argument('--seed', type=int, required=True)
    o.add_argument('--threads', type=int, default=3)
    p.add_argument('--run', action='append', default=[], metavar='STUDY:BUDGET')
    p.add_argument('--processes', type=int, default=4)
    p.add_argument('--threads', type=int, default=3)
    a = p.parse_args()
    if a.command == 'one':
        cmd_one(a)
    else:
        if not a.run:
            p.error('give at least one --run STUDY:BUDGET')
        cmd_all(a)


if __name__ == '__main__':
    main()
