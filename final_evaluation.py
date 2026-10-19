''' Final evaluation of the tuned agents: head-to-head round robin,
evaluation-panel performance and exploitability lower bounds, written as
CSV, LaTeX and Markdown tables for the thesis.

Pipeline:
  1. tune each algorithm with tune.py and run `tune.py finalists`
  2. build a manifest of the winning configuration's agents (all of its
     finalist seeds, so results can be reported as mean +/- spread over
     seeds rather than for one lucky seed):
        python final_evaluation.py manifest --study dqn=dqn_v1 --study ppo=ppo_v1 \
            --study nfsp=nfsp_v1 --study deepcfr=deepcfr_v1 [--add tag=tag]
     (or write/edit experiments/final/manifest.json by hand)
  3. run every evaluation (resumable: finished tasks are kept in results.json):
        python final_evaluation.py run [--h2h_deals 2000 --panel_deals 5000 --lbr_deals 5000]
  4. write the tables (also done at the end of `run`):
        python final_evaluation.py report

Manifest entries: {"name": "dqn_s1042", "algo": "dqn", "kind": "dqn",
"ckpt": "experiments/tune/.../finalists/trial_7/seed_1042", "seed": 1042}.
Rule-based bots can be added as reference agents (`--add tag=tag`); they
take part in the head-to-head matches, and get panel and LBR results too.

All matches use duplicate deals with fixed seeds, so every pair of agents
plays the same cards. Win rates are mbb/hand from the row agent's view.
'''

import os
import sys
import json
import math
import time
import argparse
import itertools
from collections import OrderedDict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
FINAL_ROOT = os.path.join(REPO_ROOT, 'experiments', 'final')
CHECKPOINT_KINDS = ('dqn', 'nfsp', 'deepcfr', 'ppo')
ALGO_ORDER = ['dqn', 'ppo', 'nfsp', 'deepcfr']


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def build_manifest(studies, extra):
    entries = []
    for spec in studies:
        algo, study = spec.split('=', 1)
        path = os.path.join(REPO_ROOT, 'experiments', 'tune', study, 'finalists', 'finalists.json')
        if not os.path.isfile(path):
            raise SystemExit(f'{path} not found -- run `python tune.py finalists --study {study}` first')
        with open(path, encoding='utf-8') as f:
            record = json.load(f)
        winner = record.get('winner')
        if not winner:
            raise SystemExit(f'{path} has no winner yet')
        runs = [r for r in record['runs'] if r['trial'] == winner['trial']]
        for r in runs:
            entries.append({'name': f'{algo}_s{r["seed"]}', 'algo': algo, 'kind': algo,
                            'ckpt': r['log_dir'], 'seed': r['seed'], 'study': study,
                            'trial': winner['trial'], 'params': winner['params']})
    for spec in extra:
        name, kind = spec.split('=', 1)
        kind, _, ckpt = kind.partition(':')
        entries.append({'name': name, 'algo': name, 'kind': kind, 'ckpt': ckpt or None})
    return entries


def load_manifest(path):
    with open(path, encoding='utf-8') as f:
        entries = json.load(f)
    names = [e['name'] for e in entries]
    if len(set(names)) != len(names):
        raise SystemExit('manifest names must be unique')
    return entries


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #

class Results(object):
    ''' results.json: one entry per finished task, written after each task. '''

    def __init__(self, path, settings):
        self.path = path
        self.data = {'settings': settings, 'tasks': {}}
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                old = json.load(f)
            if old.get('settings') != settings:
                raise SystemExit(f'{path} was produced with different settings {old.get("settings")}; '
                                 f'use another --out folder or the same settings')
            self.data = old

    def has(self, key):
        return key in self.data['tasks']

    def get(self, key):
        return self.data['tasks'].get(key)

    def put(self, key, value):
        self.data['tasks'][key] = value
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)


def run(args):
    from agent_loading import load_policy
    from eval_panel import Panel, DEFAULT_MEMBERS, duplicate_match
    from exploitability import LBR, combine_bounds

    entries = load_manifest(args.manifest)
    os.makedirs(args.out, exist_ok=True)
    settings = {'h2h_deals': args.h2h_deals, 'panel_deals': args.panel_deals,
                'lbr_deals': args.lbr_deals, 'lbr_samples': args.lbr_samples}
    results = Results(os.path.join(args.out, 'results.json'), settings)

    policies = {}

    def policy(e):
        if e['name'] not in policies:
            ckpt = e.get('ckpt')
            if ckpt and not os.path.isabs(ckpt):
                ckpt = os.path.join(REPO_ROOT, ckpt)
            policies[e['name']] = load_policy(e['kind'], ckpt if e['kind'] in CHECKPOINT_KINDS else None)
        return policies[e['name']]

    tasks = []
    for a, b in itertools.combinations(entries, 2):
        if a['algo'] != b['algo']:                 # seeds of one algorithm are not matched up
            tasks.append(('h2h', a, b))
    for e in entries:
        tasks.append(('panel', e, None))
        tasks.append(('lbr', e, None))
    todo = [t for t in tasks if not results.has(_key(*t))]
    print(f'{len(tasks)} tasks, {len(tasks) - len(todo)} already done', flush=True)

    for i, (kind, a, b) in enumerate(todo, 1):
        start = time.perf_counter()
        key = _key(kind, a, b)
        if kind == 'h2h':
            r = duplicate_match(policy(a), policy(b), args.h2h_deals)
            text = f'{a["name"]} vs {b["name"]}: {r["mean"]:+.1f} +/- {r["se"]:.1f}'
        elif kind == 'panel':
            p = Panel(DEFAULT_MEMBERS, num_deals=args.panel_deals).evaluate(policy(a))
            r = {'members': p['members'], 'min': p['min'], 'mean': p['mean'], 'actions': p['actions']}
            text = f'{a["name"]} panel min {r["min"]:+.1f}'
        else:
            r = LBR(policy(a), samples=args.lbr_samples).evaluate(args.lbr_deals)
            text = f'{a["name"]} LBR {r["mean"]:+.1f} +/- {r["se"]:.1f}'
        r['seconds'] = time.perf_counter() - start
        results.put(key, r)
        print(f'[{i}/{len(todo)}] {kind:5s} {text}  ({r["seconds"]:.0f}s)', flush=True)
    report(argparse.Namespace(manifest=args.manifest, out=args.out))


def _key(kind, a, b=None):
    return f'{kind}:{a["name"]}' + (f':{b["name"]}' if b else '')


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def _pm(mean, err, digits=0):
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return 'n/a'
    return f'{mean:+.{digits}f} ± {err:.{digits}f}' if err is not None else f'{mean:+.{digits}f}'


def _tex(text):
    return text.replace('±', r'$\pm$').replace('_', r'\_').replace('%', r'\%')


def _write_table(path_base, title, header, rows, note=''):
    import csv
    with open(path_base + '.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(path_base + '.tex', 'w', encoding='utf-8') as f:
        f.write('% ' + title + '\n')
        f.write('\\begin{tabular}{l' + 'r' * (len(header) - 1) + '}\n\\toprule\n')
        f.write(' & '.join(_tex(h) for h in header) + ' \\\\\n\\midrule\n')
        for row in rows:
            f.write(' & '.join(_tex(str(c)) for c in row) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
    md = [f'### {title}', '', '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    md += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    if note:
        md += ['', note]
    return '\n'.join(md) + '\n'


def report(args):
    entries = load_manifest(args.manifest)
    path = os.path.join(args.out, 'results.json')
    if not os.path.isfile(path):
        raise SystemExit(f'{path} not found -- run `final_evaluation.py run` first')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    tasks, settings = data['tasks'], data['settings']
    names = [e['name'] for e in entries]
    algos = []
    for e in entries:
        if e['algo'] not in algos:
            algos.append(e['algo'])
    algos.sort(key=lambda a: ALGO_ORDER.index(a) if a in ALGO_ORDER else len(ALGO_ORDER))
    by_algo = {a: [e for e in entries if e['algo'] == a] for a in algos}
    md = [f'# Final evaluation\n\nSettings: {settings}. All values in mbb/hand; ± is one standard error '
          f'(per agent) or the standard deviation over seeds (per algorithm, "sd").\n']

    # ---- head-to-head, agents
    def h2h(a, b):
        r = tasks.get(f'h2h:{a}:{b}')
        if r:
            return r['mean'], r['se']
        r = tasks.get(f'h2h:{b}:{a}')
        if r:
            return -r['mean'], r['se']
        return None
    rows = []
    for a in names:
        row = [a]
        for b in names:
            v = h2h(a, b) if a != b else None
            row.append('--' if a == b else (_pm(*v) if v else ''))
        rows.append(row)
    md.append(_write_table(os.path.join(args.out, 'h2h_agents'), 'Head-to-head, every agent (row vs column)',
                           ['agent'] + names, rows))

    # ---- head-to-head, algorithms: mean over all seed pairs
    rows = []
    for a in algos:
        row = [a]
        for b in algos:
            if a == b:
                row.append('--')
                continue
            vals = [h2h(x['name'], y['name']) for x in by_algo[a] for y in by_algo[b]]
            vals = [v for v in vals if v]
            if not vals:
                row.append('')
                continue
            means = np.array([v[0] for v in vals])
            se = math.sqrt(sum(v[1] ** 2 for v in vals)) / len(vals)
            row.append(_pm(means.mean(), se) + (f' (sd {means.std(ddof=1):.0f})' if len(vals) > 1 else ''))
        rows.append(row)
    md.append(_write_table(os.path.join(args.out, 'h2h_algorithms'),
                           'Head-to-head by algorithm (mean over all seed pairs)', ['algorithm'] + algos, rows,
                           'Each cell averages every agent of the row algorithm against every agent of the '
                           'column algorithm; ± is the standard error of that mean, sd the spread over seed pairs.'))

    # ---- panel + exploitability, agents
    from exploitability import combine_bounds
    members = None
    agent_rows, per_agent = [], {}
    for e in entries:
        panel = tasks.get(f'panel:{e["name"]}')
        lbr = tasks.get(f'lbr:{e["name"]}')
        if members is None and panel:
            members = list(panel['members'])
        bounds = combine_bounds(lbr, panel['members']) if (panel and lbr) else None
        per_agent[e['name']] = (panel, lbr, bounds)
    members = members or []
    header = (['agent', 'panel min'] + [f'vs {m}' for m in members]
              + ['LBR', 'best lower bound', 'by', 'conservative'])
    for e in entries:
        panel, lbr, bounds = per_agent[e['name']]
        row = [e['name'], f'{panel["min"]:+.0f}' if panel else '']
        row += [_pm(panel['members'][m]['mean'], panel['members'][m]['se']) if panel else '' for m in members]
        row += [_pm(lbr['mean'], lbr['se']) if lbr else '']
        row += ([_pm(bounds['best']['mean'], bounds['best']['se']), bounds['best']['method'],
                 f'{bounds["conservative"]["value"]:.0f}'] if bounds else ['', '', ''])
        agent_rows.append(row)
    md.append(_write_table(os.path.join(args.out, 'agents'),
                           'Panel performance and exploitability lower bounds, every agent', header, agent_rows,
                           'Panel columns: the agent\'s win rate against each rule-based bot (panel min = '
                           'the worst of them, excluding random). LBR / best lower bound: what a '
                           'counter-strategy wins against the agent (higher = more exploitable); '
                           'conservative = largest 95% lower confidence bound, floored at 0.'))

    # ---- algorithm summary over seeds
    header = ['algorithm', 'agents', 'panel min'] + [f'vs {m}' for m in members] + ['LBR', 'best lower bound']
    rows = []
    for a in algos:
        stats = [per_agent[e['name']] for e in by_algo[a]]
        stats = [s for s in stats if s[0] and s[1]]
        if not stats:
            continue

        def agg(values):
            values = np.array(values, dtype=float)
            sd = values.std(ddof=1) if len(values) > 1 else float('nan')
            return f'{values.mean():+.0f}' + (f' (sd {sd:.0f})' if len(values) > 1 else '')
        row = [a, len(stats), agg([s[0]['min'] for s in stats])]
        row += [agg([s[0]['members'][m]['mean'] for s in stats]) for m in members]
        row += [agg([s[1]['mean'] for s in stats]), agg([s[2]['best']['mean'] for s in stats])]
        rows.append(row)
    md.append(_write_table(os.path.join(args.out, 'algorithms'),
                           'Summary by algorithm (mean over seeds, sd = spread over seeds)', header, rows))

    with open(os.path.join(args.out, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    missing = sum(1 for e in entries for k in ('panel', 'lbr') if f'{k}:{e["name"]}' not in tasks)
    print(f'tables written to {args.out} (h2h_agents, h2h_algorithms, agents, algorithms: .csv/.tex; summary.md)'
          + (f'; {missing} panel/LBR task(s) still missing' if missing else ''))


def main():
    p = argparse.ArgumentParser('Final evaluation: round robin + panel + exploitability')
    sub = p.add_subparsers(dest='command', required=True)
    m = sub.add_parser('manifest', help='build the agent manifest from tuning studies')
    m.add_argument('--study', action='append', default=[], help='algo=study_name (repeatable)')
    m.add_argument('--add', action='append', default=[],
                   help='extra agent name=kind[:checkpoint], e.g. tag=tag or mydqn=dqn:experiments/x')
    m.add_argument('--out', default=os.path.join(FINAL_ROOT, 'manifest.json'))
    r = sub.add_parser('run', help='run every evaluation (resumable)')
    r.add_argument('--manifest', default=os.path.join(FINAL_ROOT, 'manifest.json'))
    r.add_argument('--out', default=FINAL_ROOT)
    r.add_argument('--h2h_deals', type=int, default=2000, help='duplicate deals per head-to-head pair')
    r.add_argument('--panel_deals', type=int, default=5000, help='duplicate deals per panel member')
    r.add_argument('--lbr_deals', type=int, default=5000)
    r.add_argument('--lbr_samples', type=int, default=500)
    g = sub.add_parser('report', help='write the tables from results.json')
    g.add_argument('--manifest', default=os.path.join(FINAL_ROOT, 'manifest.json'))
    g.add_argument('--out', default=FINAL_ROOT)
    a = p.parse_args()

    if a.command == 'manifest':
        entries = build_manifest(a.study, a.add)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2)
        print(f'{len(entries)} agents -> {a.out}')
        for e in entries:
            print(f'  {e["name"]:16s} {e["kind"]:8s} {e.get("ckpt") or ""}')
    elif a.command == 'run':
        run(a)
    else:
        report(a)


if __name__ == '__main__':
    main()
