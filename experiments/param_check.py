''' Does one parameter explain a behaviour? Re-train a study's best trial with
a parameter set to a few values (several seeds each, full budget, no early
stopping) and evaluate every run on held-out deals (heldout_eval.py).

    python experiments/param_check.py run --study nfsp_v2 --param q_epsilon_start \
        --values 0.15,0.25 --seeds 2 --workers 4 [--wait]
    python experiments/param_check.py report --study nfsp_v2 --param q_epsilon_start

The trial's own value is always included as the baseline. Runs live in
experiments/param_check/<study>_<param>/<value>/seed_<S>/ (checkpoint,
training logs, result.json); `run` is resumable. --wait first waits until the
study has finished (all trials done, no worker left).
'''
import os
import sys
import csv
import json
import math
import time
import shutil
import argparse
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
ROOT = os.path.join(REPO, 'experiments', 'param_check')


def best_trial(study):
    import tune
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    s = tune.load_study(study)
    return max((t for t in s.trials if t.state.name == 'COMPLETE'), key=lambda t: t.value)


def out_dir(study, param, value, seed):
    return os.path.join(ROOT, f'{study}_{param}', str(value), f'seed_{seed}')


def cmd_train(a):
    import torch
    import tune
    from heldout_eval import evaluate
    torch.set_num_threads(a.threads)
    cfg = tune.load_config(a.study)
    t = best_trial(a.study)
    params = dict(t.params)
    value = json.loads(a.value) if a.value != 'baseline' else params[a.param]
    params[a.param] = value
    d = out_dir(a.study, a.param, a.value, a.seed)
    module, args = tune.build_args(cfg, params, d, a.seed, save_model=True)
    t0 = time.time()
    module.train(args, None)
    train_seconds = time.time() - t0
    held = evaluate(cfg['algo'], d, a.panel_deals, a.lbr_deals, a.play_deals, name=f'{a.param}={value}')
    with open(os.path.join(d, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump({'study': a.study, 'trial': t.number, 'param': a.param, 'value': value, 'label': a.value,
                   'seed': a.seed, 'params': params, 'train_seconds': train_seconds, **held}, f, indent=1)


def study_finished(study):
    from study_runner import study_state, workers_alive
    _, finished, running = study_state(study)
    return running == 0 and workers_alive(study) == 0, finished


def cmd_run(a):
    if a.wait:
        while True:
            done, finished = study_finished(a.study)
            if done:
                break
            print(time.strftime('%H:%M'), f'waiting for {a.study} to finish ({finished} trials finished)', flush=True)
            time.sleep(300)
    import tune
    cfg = tune.load_config(a.study)
    t = best_trial(a.study)
    labels = ['baseline'] + [v.strip() for v in a.values.split(',') if v.strip()]
    seeds = [cfg['seed'] + 3000 + i for i in range(a.seeds)]
    print(f'{a.study} trial {t.number} (score {t.value:+.0f}), {a.param} = {t.params[a.param]} (baseline) vs '
          f'{labels[1:]}, seeds {seeds}', flush=True)
    queue = [(lab, s) for s in seeds for lab in labels
             if not os.path.isfile(os.path.join(out_dir(a.study, a.param, lab, s), 'result.json'))]
    running = []
    while queue or running:
        while queue and len(running) < a.workers:
            lab, s = queue.pop(0)
            d = out_dir(a.study, a.param, lab, s)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
            cmd = [sys.executable, '-u', os.path.abspath(__file__), 'train', '--study', a.study, '--param', a.param,
                   '--value', lab, '--seed', str(s), '--threads', str(a.threads), '--panel_deals',
                   str(a.panel_deals), '--lbr_deals', str(a.lbr_deals), '--play_deals', str(a.play_deals)]
            log = open(os.path.join(d, 'run.log'), 'w', encoding='utf-8')
            running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO), lab, s, log,
                            time.time()))
            print(time.strftime('%H:%M'), f'started {a.param}={lab} seed {s}', flush=True)
        time.sleep(15)
        for item in list(running):
            proc, lab, s, log, t0 = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                print(time.strftime('%H:%M'), f'{a.param}={lab} seed {s}: '
                      + ('finished' if proc.returncode == 0 else f'FAILED (exit {proc.returncode})')
                      + f' after {(time.time() - t0) / 60:.0f} min', flush=True)
    cmd_report(a)
    print('param check done', flush=True)


def _agg(xs, pct=False):
    xs = [x for x in xs if x is not None]
    if not xs:
        return ''
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
    return (f'{m:.0%}' + (f' (sd {sd:.0%})' if sd is not None else '')) if pct else \
        (f'{m:+.0f}' + (f' (sd {sd:.0f})' if sd is not None else ''))


def cmd_report(a):
    base = os.path.join(ROOT, f'{a.study}_{a.param}')
    rows = []
    for lab in sorted(os.listdir(base), key=lambda x: (x != 'baseline', x)) if os.path.isdir(base) else []:
        rs = []
        for s in sorted(os.listdir(os.path.join(base, lab))):
            p = os.path.join(base, lab, s, 'result.json')
            if os.path.isfile(p):
                rs.append((json.load(open(p, encoding='utf-8')), os.path.join(base, lab, s)))
        if not rs:
            continue
        curves = []
        for r, d in rs:
            fold = [float(x['act_fold']) for x in csv.DictReader(open(os.path.join(d, 'panel.csv')))]
            curves.append(' '.join(f'{f * 100:.1f}' for f in fold))

        def play(r, k):
            return r['play_vs_tag'][k]['value']
        res = [r for r, _ in rs]
        rows.append([f'{lab} ({res[0]["value"]})', len(rs), _agg([r['panel']['min'] for r in res]),
                     _agg([r['lbr']['mean'] for r in res])]
                    + [_agg([r['panel']['members'][m]['mean'] for r in res]) for m in ('station', 'maniac', 'nit', 'tag', 'lag')]
                    + [_agg([play(r, k) for r in res], pct=True) for k in
                       ('fold facing a bet', 'fold when equity < pot odds', 'fold when equity >> pot odds', 'VPIP', 'PFR')]
                    + [' / '.join(curves)])
    header = ['value', 'seeds', 'held-out panel min', 'LBR (lower = better)', 'vs station', 'vs maniac', 'vs nit',
              'vs tag', 'vs lag', 'fold facing bet', 'fold when behind', 'fold when well ahead', 'VPIP', 'PFR',
              'training fold % per evaluation (per seed)']
    lines = [f'# {a.study}: {a.param}', '', 'mbb/hand; mean over seeds (sd = spread over seeds); held-out deals.', '',
             '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r, t, g = sub.add_parser('run'), sub.add_parser('train'), sub.add_parser('report')
    for q in (r, t, g):
        q.add_argument('--study', required=True)
        q.add_argument('--param', required=True)
    for q in (r, t):
        q.add_argument('--threads', type=int, default=3)
        q.add_argument('--panel_deals', type=int, default=2000)
        q.add_argument('--lbr_deals', type=int, default=1000)
        q.add_argument('--play_deals', type=int, default=500)
    r.add_argument('--values', required=True)
    r.add_argument('--seeds', type=int, default=2)
    r.add_argument('--workers', type=int, default=4)
    r.add_argument('--wait', action='store_true')
    t.add_argument('--value', required=True)
    t.add_argument('--seed', type=int, required=True)
    a = p.parse_args()
    {'run': cmd_run, 'train': cmd_train, 'report': cmd_report}[a.command](a)


if __name__ == '__main__':
    main()
