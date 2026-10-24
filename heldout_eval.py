''' Held-out evaluation of saved agents, on deals that neither tuning nor
`tune.py finalists` (which picks the winner) ever used:
  panel  the evaluation panel on final_evaluation's held-out deal seed
  LBR    local best response exploitability (lower = less exploitable)
  play   duplicate deals against TAG recorded with behaviour_tracker:
         VPIP / PFR / aggression and fold decisions relative to pot odds

    python heldout_eval.py finalists --study ppo_v1 --study ppo_v2 [--workers 4]
        evaluates every finalist run of the studies (heldout.json next to each
        checkpoint; finished ones are skipped) and prints a comparison
    python heldout_eval.py table --study ppo_v1 --study ppo_v2
'''

import os
import sys
import json
import math
import time
import argparse
import subprocess

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)


def fold_stats(hands, role='agent'):
    ''' Fold behaviour relative to pot odds, over decisions facing a bet
    where folding was legal; 'equity' is the hand's equity against a random
    hand (no hindsight). '''
    behind, ahead, facing = [], [], []
    for h in hands:
        for d in h['decisions']:
            if d['role'] != role or d['to_call_bb'] <= 0 or 'fold' not in d['legal']:
                continue
            p = d['probs']['fold'] if d['probs'] else float(d['action'] == 'fold')
            need = d['to_call_bb'] / (d['pot_bb'] + d['to_call_bb'])
            facing.append(p)
            if d['equity'] < need - 0.05:
                behind.append(p)
            elif d['equity'] > need + 0.15:
                ahead.append(p)

    def mean(xs):
        return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)
    return {'fold facing a bet': mean(facing),
            'fold when equity < pot odds': mean(behind),
            'fold when equity >> pot odds': mean(ahead)}


def evaluate(kind, ckpt, panel_deals=2000, lbr_deals=1000, play_deals=500, name='agent'):
    from agent_loading import load_policy
    from eval_panel import Panel, DEFAULT_MEMBERS
    from exploitability import LBR
    from final_evaluation import FINAL_PANEL_SEED
    import behaviour_tracker as bt
    policy = load_policy(kind, ckpt)
    panel = Panel(DEFAULT_MEMBERS, num_deals=panel_deals, eval_seed=FINAL_PANEL_SEED).evaluate(policy)
    lbr = LBR(policy, samples=500).evaluate(lbr_deals)
    hands = bt.play_match(policy, load_policy('tag'), play_deals, names=(name, 'tag'))
    stats = bt.stats_to_json(bt.compute_stats(hands))
    stats.update({k: {'value': v[0], 'n': v[1]} for k, v in fold_stats(hands).items()})
    return {'panel': {'min': panel['min'], 'mean': panel['mean'], 'members': panel['members'],
                      'actions': panel['actions']},
            'lbr': {'mean': lbr['mean'], 'se': lbr['se'], 'deals': lbr['deals']},
            'play_vs_tag': stats,
            'settings': {'panel_deals': panel_deals, 'lbr_deals': lbr_deals, 'play_deals': play_deals}}


def _finalist_runs(study):
    path = os.path.join(REPO_ROOT, 'experiments', 'tune', study, 'finalists', 'finalists.json')
    if not os.path.isfile(path):
        raise SystemExit(f'{path} not found -- run `python tune.py finalists --study {study}` first')
    with open(path, encoding='utf-8') as f:
        record = json.load(f)
    return record, [r for r in record['runs'] if 'failed' not in r]


def cmd_one(a):
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    out = evaluate(a.kind, a.dir, a.panel_deals, a.lbr_deals, a.play_deals)
    with open(os.path.join(a.dir, 'heldout.json.tmp'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    os.replace(os.path.join(a.dir, 'heldout.json.tmp'), os.path.join(a.dir, 'heldout.json'))
    print(f'{a.dir}: held-out panel min {out["panel"]["min"]:+.0f}, LBR {out["lbr"]["mean"]:+.0f}', flush=True)


def cmd_finalists(a):
    import tune
    queue = []
    for study in a.study:
        kind = tune.load_config(study)['algo']
        _, runs = _finalist_runs(study)
        for r in runs:
            d = os.path.join(REPO_ROOT, r['log_dir'])
            if not os.path.isfile(os.path.join(d, 'heldout.json')):
                queue.append((kind, d))
    print(f'{len(queue)} finalist run(s) to evaluate', flush=True)
    running = []
    while queue or running:
        while queue and len(running) < a.workers:
            kind, d = queue.pop(0)
            cmd = [sys.executable, '-u', os.path.abspath(__file__), 'one', '--kind', kind, '--dir', d,
                   '--threads', str(a.threads), '--panel_deals', str(a.panel_deals),
                   '--lbr_deals', str(a.lbr_deals), '--play_deals', str(a.play_deals)]
            log = open(os.path.join(d, 'heldout.log'), 'w', encoding='utf-8')
            running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT), d, log))
        time.sleep(5)
        for item in list(running):
            proc, d, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                print(f'  {os.path.relpath(d, REPO_ROOT)}: '
                      + ('done' if proc.returncode == 0 else f'FAILED (see heldout.log)'), flush=True)
    cmd_table(a)


def _agg(values, pct=False):
    values = [v for v in values if v is not None]
    if not values:
        return ''
    m = sum(values) / len(values)
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else None
    if pct:
        return f'{m:.0%}' + (f' (sd {sd:.0%})' if sd is not None else '')
    return f'{m:+.0f}' + (f' (sd {sd:.0f})' if sd is not None else '')


def cmd_table(a):
    lines = ['| study | trial | seeds | tuning score | finalists panel min (tuning deals) | held-out panel min | '
             'LBR (lower = better) | vs station | vs maniac | vs nit | vs tag | vs lag | fold facing bet | '
             'fold when behind | fold when well ahead | VPIP | PFR |',
             '|' + '---|' * 17]
    for study in a.study:
        record, runs = _finalist_runs(study)
        winner = (record.get('winner') or {}).get('trial')
        by_trial = {}
        for r in runs:
            p = os.path.join(REPO_ROOT, r['log_dir'], 'heldout.json')
            if os.path.isfile(p):
                with open(p, encoding='utf-8') as f:
                    by_trial.setdefault(r['trial'], []).append((r, json.load(f)))
        for trial, items in sorted(by_trial.items(), key=lambda kv: -kv[1][0][0]['trial_score']):
            h = [x[1] for x in items]

            def play(k):
                return [x['play_vs_tag'][k]['value'] for x in h]
            lines.append('| ' + ' | '.join(str(c) for c in [
                study, f'{trial}' + (' (winner)' if trial == winner else ''), len(items),
                f'{items[0][0]["trial_score"]:+.0f}', _agg([x[0]['final_min'] for x in items]),
                _agg([x['panel']['min'] for x in h]), _agg([x['lbr']['mean'] for x in h])]
                + [_agg([x['panel']['members'][m]['mean'] for x in h]) for m in ('station', 'maniac', 'nit', 'tag', 'lag')]
                + [_agg(play(k), pct=True) for k in ('fold facing a bet', 'fold when equity < pot odds',
                                                     'fold when equity >> pot odds', 'VPIP', 'PFR')]) + ' |')
    text = '\n'.join(lines)
    print(text)
    out = os.path.join(REPO_ROOT, 'experiments', 'heldout_' + '_vs_'.join(a.study) + '.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('# Held-out comparison of finalists\n\nmbb/hand; mean over seeds (sd = spread over seeds). '
                'Held-out panel and LBR use deals never used for tuning or for picking the winner; fold rates '
                'from duplicate deals vs TAG ("behind": equity below the pot odds, "well ahead": 15 points '
                'above).\n\n' + text + '\n')
    print(f'\nwritten to {out}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    o = sub.add_parser('one')
    o.add_argument('--kind', required=True)
    o.add_argument('--dir', required=True)
    f = sub.add_parser('finalists')
    f.add_argument('--workers', type=int, default=4)
    for q in (o, f):
        q.add_argument('--threads', type=int, default=3)
        q.add_argument('--panel_deals', type=int, default=2000)
        q.add_argument('--lbr_deals', type=int, default=1000)
        q.add_argument('--play_deals', type=int, default=500)
    for q in (f, sub.add_parser('table')):
        q.add_argument('--study', action='append', required=True)
    a = p.parse_args()
    {'one': cmd_one, 'finalists': cmd_finalists, 'table': cmd_table}[a.command](a)


if __name__ == '__main__':
    main()
