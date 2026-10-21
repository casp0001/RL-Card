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
# Held-out panel deals: tuning trials and `tune.py finalists` (which picks the
# winner) use the panel's default deals from eval_panel.EVAL_SEED, so the
# reported panel results use different deals to avoid selection bias.
FINAL_PANEL_SEED = 50_000_000
LBR_CHUNK = 250     # LBR deals per saved chunk (rule-based agents take ~13 s per deal)


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
        runs = [r for r in record['runs'] if r['trial'] == winner['trial'] and 'failed' not in r]
        expected = winner.get('seeds')
        if expected and sorted(r['seed'] for r in runs) != sorted(expected):
            raise SystemExit(f'{path}: the winner should have seeds {expected} but has '
                             f'{sorted(r["seed"] for r in runs)} -- finish `tune.py finalists` first')
        for r in runs:
            entries.append({'name': f'{algo}_s{r["seed"]}', 'algo': algo, 'kind': algo,
                            'ckpt': r['log_dir'].replace('\\', '/'), 'seed': r['seed'], 'study': study,
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
        self._save()

    def drop(self, key):
        if self.data['tasks'].pop(key, None) is not None:
            self._save()

    def _save(self):
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
                'lbr_deals': args.lbr_deals, 'lbr_bot_deals': args.lbr_bot_deals,
                'lbr_samples': args.lbr_samples, 'panel_seed': FINAL_PANEL_SEED}
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
            p = Panel(DEFAULT_MEMBERS, num_deals=args.panel_deals, eval_seed=FINAL_PANEL_SEED).evaluate(policy(a))
            r = {'members': p['members'], 'min': p['min'], 'mean': p['mean'], 'actions': p['actions']}
            text = f'{a["name"]} panel min {r["min"]:+.1f}'
        else:
            deals = args.lbr_deals if a['kind'] in CHECKPOINT_KINDS else args.lbr_bot_deals
            r = _lbr_chunked(results, key, policy(a), deals, args.lbr_samples, LBR, a['name'])
            text = f'{a["name"]} LBR {r["mean"]:+.1f} +/- {r["se"]:.1f}'
        r['seconds'] = r.get('seconds_total', time.perf_counter() - start)
        results.put(key, r)
        print(f'[{i}/{len(todo)}] {kind:5s} {text}  ({r["seconds"]:.0f}s)', flush=True)
    report(argparse.Namespace(manifest=args.manifest, out=args.out))


def _key(kind, a, b=None):
    return f'{kind}:{a["name"]}' + (f':{b["name"]}' if b else '')


def _lbr_chunked(results, key, policy, num_deals, samples, LBR, name):
    ''' LBR in chunks of LBR_CHUNK deals; each finished chunk is saved under
    'partial:<key>', so an interruption loses at most one chunk. '''
    from exploitability import summarize_deals
    partial_key = 'partial:' + key
    part = results.get(partial_key) or {'per_deal': [], 'seconds': 0.0, 'inconsistent_actions': 0}
    lbr = LBR(policy, samples=samples)
    while len(part['per_deal']) < num_deals:
        done = len(part['per_deal'])
        n = min(LBR_CHUNK, num_deals - done)
        before = lbr.inconsistent
        r = lbr.evaluate(n, first_deal=done, return_per_deal=True)
        part['per_deal'] += r['per_deal']
        part['seconds'] += r['seconds']
        part['inconsistent_actions'] += lbr.inconsistent - before
        results.put(partial_key, part)
        s = summarize_deals(part['per_deal'])
        print(f'    {name} LBR {s["deals"]}/{num_deals} deals: {s["mean"]:+.1f} +/- {s["se"]:.1f} '
              f'({part["seconds"]:.0f}s)', flush=True)
    out = summarize_deals(part['per_deal'])
    out.update(seconds_total=part['seconds'], inconsistent_actions=part['inconsistent_actions'])
    results.drop(partial_key)
    return out


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


def _h2h_algo_se(grid, deal_var):
    ''' Standard error of the mean of a (row seeds x column seeds) grid of
    head-to-head results: a two-way random-effects approximation, i.e. the
    seed-to-seed variation of the row agents' means and of the column agents'
    means, plus the deal noise of the matches themselves. '''
    def seed_term(means):
        means = means[np.isfinite(means)]
        return means.var(ddof=1) / len(means) if len(means) > 1 else 0.0
    with np.errstate(all='ignore'):
        rows, cols = np.nanmean(grid, axis=1), np.nanmean(grid, axis=0)
    deal = sum(deal_var) / len(deal_var) ** 2
    return math.sqrt(seed_term(rows) + seed_term(cols) + deal)


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
            grid = np.full((len(by_algo[a]), len(by_algo[b])), np.nan)
            deal_var = []
            for i, x in enumerate(by_algo[a]):
                for j, y in enumerate(by_algo[b]):
                    v = h2h(x['name'], y['name'])
                    if v:
                        grid[i, j] = v[0]
                        deal_var.append(v[1] ** 2)
            if not deal_var:
                row.append('')
                continue
            row.append(_pm(float(np.nanmean(grid)), _h2h_algo_se(grid, deal_var))
                       + (f' (sd {np.nanstd(grid, ddof=1):.0f})' if np.isfinite(grid).sum() > 1 else ''))
        rows.append(row)
    md.append(_write_table(os.path.join(args.out, 'h2h_algorithms'),
                           'Head-to-head by algorithm (mean over all seed pairs)', ['algorithm'] + algos, rows,
                           'Each cell averages every agent of the row algorithm against every agent of the '
                           'column algorithm; ± is a standard error that counts the training seeds of both '
                           'algorithms as the random units (seed pairs share agents, so they are not '
                           'independent) plus the card noise; sd is the spread over seed pairs.'))

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
    r.add_argument('--lbr_bot_deals', type=int, default=500,
                   help='LBR deals for rule-based reference agents (--add tag=tag): ~13 s per deal')
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
