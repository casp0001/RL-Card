''' Thesis material that needs no new training, written to experiments/final/:

  curves       learning curves of the winners (mean over seeds, band = range):
               limit hold'em (training-time panel min, standard and extended
               runs) and Leduc (exact exploitability), against the training
               budget and against wall time            -> figures/*.png / *.pdf / *.csv
  cost         training cost: budget, wall time per run, network size and
               tuning effort per algorithm             -> cost.csv / .tex (+ summary in cost.md)
  leduc_h2h    exact Leduc round robin of the winners' seeds (no card noise),
               against the CFR+ equilibrium too, and how it ranks the
               algorithms compared with exploitability -> leduc_h2h*.csv / .tex, leduc_h2h.md
  tuning       per study: search space, winner, parameter importance (fANOVA)
               and the optimization history         -> tuning/tuning.md, <study>.csv / .tex, *_history.png
  nfsp_divergence  the diverged NFSP studies v1 / v2 against the fixed v3
                                                       -> nfsp_divergence.md / .csv / .tex
  hands        each winner against tag: characteristic hands (bluffs, slow
               plays, biggest wins and losses) as text and HTML replays, with
               the mistake measure of every decision (hand_analysis.py: EV loss
               against tag by rollouts; Leduc: exact, against CFR+) and the
               viewer page                          -> hands/<algo>_vs_tag.*, leduc_<algo>_vs_cfr.*, hands.html
  provenance   git commit, uncommitted files, library versions, settings
                                                       -> provenance.json

    python final_report.py all            (or one of the commands above)

Wall time per evaluation point is interpolated linearly between a run's start
and end (Deep CFR slows down as its memories fill, so its early points are
slightly too late on the time axis).
'''

import os
import sys
import json
import time
import hashlib
import argparse
import platform
import statistics
import subprocess
from collections import OrderedDict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TUNE = os.path.join(REPO_ROOT, 'experiments', 'tune')
OUT = os.path.join(REPO_ROOT, 'experiments', 'final')

# algorithm -> (limit study, Leduc study, checkpoint file)
ALGOS = OrderedDict([
    ('dqn', ('dqn_v1', 'leduc_dqn_v1', 'checkpoint_dqn.pt')),
    ('ppo', ('ppo_v2', 'leduc_ppo_v1', 'checkpoint_ppo.pt')),
    ('nfsp', ('nfsp_v3', 'leduc_nfsp_v1', 'checkpoint_nfsp.pt')),
    ('deepcfr', ('deepcfr_v1', 'leduc_deepcfr_v1', 'model.ckpt')),
])
EXTENDED = {'ppo': 4_000_000, 'nfsp': 8_000_000, 'deepcfr': 500}
UNIT = {'dqn': 'episodes', 'ppo': 'episodes', 'nfsp': 'episodes', 'deepcfr': 'iterations'}
LABEL = {'dqn': 'DQN', 'ppo': 'PPO', 'nfsp': 'NFSP', 'deepcfr': 'Deep CFR'}
# validated categorical palette (dataviz validate_palette.js, light surface): blue, yellow, aqua, red;
# the CVD separation of red/aqua is in the 6-8 band, so every series also has its own marker and a direct label
COLOR = {'dqn': '#2a78d6', 'ppo': '#eda100', 'nfsp': '#1baf7a', 'deepcfr': '#e34948'}
MARKER = {'dqn': 'o', 'ppo': 's', 'nfsp': '^', 'deepcfr': 'D'}
INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #

def winner(study):
    with open(os.path.join(TUNE, study, 'finalists', 'finalists.json'), encoding='utf-8') as f:
        return json.load(f)['winner']


def config(study):
    with open(os.path.join(TUNE, study, 'config.json'), encoding='utf-8') as f:
        return json.load(f)


def runs(study, extended_budget=None):
    ''' Run folders of a study's winner (finalists, or its extended runs). '''
    w = winner(study)
    sub = f'extended_{extended_budget}' if extended_budget else 'finalists'
    dirs = [os.path.join(TUNE, study, sub, f'trial_{w["trial"]}', f'seed_{s}') for s in w['seeds']]
    return [d for d in dirs if os.path.isfile(os.path.join(d, 'panel.csv'))]


def curve(d):
    ''' (training steps, panel min) of one run's evaluations. '''
    data = np.genfromtxt(os.path.join(d, 'panel.csv'), delimiter=',', names=True)
    data = np.atleast_1d(data)
    return data['episode'] + 1, data['min']


def wall_hours(d):
    ''' Training wall time of a run in hours (None if unknown). '''
    ext = os.path.join(d, 'extended.json')
    if os.path.isfile(ext):
        with open(ext, encoding='utf-8') as f:
            return json.load(f)['hours']
    claim = os.path.join(d, 'claimed_by.txt')
    if os.path.isfile(claim):
        return (os.path.getmtime(os.path.join(d, 'panel.csv')) - os.path.getmtime(claim)) / 3600
    return None


def mean_curve(dirs, budget, anchor=None):
    ''' Mean / min / max over seeds at the evaluation points all seeds share,
    and the mean wall time of each point (linear in the training progress).
    anchor = (steps, hours) of the standard runs: an extended run trains
    exactly like them up to their budget, so its time is interpolated from
    the standard runs up to there and linearly from there to its own end. '''
    curves = [curve(d) for d in dirs]
    common = sorted(set.intersection(*[set(x.tolist()) for x, _ in curves]))
    ys = np.array([[dict(zip(x.tolist(), y.tolist()))[c] for c in common] for x, y in curves])
    hours = [h for h in map(wall_hours, dirs) if h is not None]
    x = np.array(common)
    t = None
    if hours:
        end = statistics.fmean(hours)
        knots = ([0, anchor[0], budget], [0, anchor[1], end]) if anchor and anchor[1] < end else ([0, budget], [0, end])
        t = np.interp(x, *knots)
    return x, ys.mean(0), ys.min(0), ys.max(0), t


def _standard_anchor(study):
    hours = [h for h in map(wall_hours, runs(study)) if h is not None]
    return (config(study)['budget'], statistics.fmean(hours)) if hours else None


def _plain_log_axis(axis):
    import matplotlib.ticker as ticker
    axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    axis.set_minor_formatter(ticker.FuncFormatter(lambda v, _: f'{v:g}' if f'{v:g}'.lstrip('0.')[:1] in '25' else ''))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _axes_style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c3c2b7')
    ax.tick_params(which='both', colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _series(ax, x, mean, lo, hi, algo, label, dashed=False, end_label=True):
    style = '--' if dashed else '-'
    ax.fill_between(x, lo, hi, color=COLOR[algo], alpha=0.15, linewidth=0)
    ax.plot(x, mean, style, color=COLOR[algo], linewidth=2, marker=MARKER[algo], markersize=4,
            markevery=max(1, len(x) // 8), label=label, markeredgecolor=SURFACE, markeredgewidth=0.8)
    if end_label:
        ax.annotate(label, (x[-1], mean[-1]), xytext=(4, 0), textcoords='offset points',
                    fontsize=7, color=INK2, va='center')


def _save(fig, name, rows, header):
    import csv
    path = os.path.join(OUT, 'figures')
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, name + '.png'), dpi=200, bbox_inches='tight', facecolor=SURFACE)
    fig.savefig(os.path.join(path, name + '.pdf'), bbox_inches='tight', facecolor=SURFACE)
    with open(os.path.join(path, name + '.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f'  figures/{name}.png / .pdf / .csv')


def cmd_curves(a):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    header = ['game', 'series', 'steps', 'hours', 'mean', 'min_over_seeds', 'max_over_seeds', 'seeds']

    # ---- limit hold'em: one panel per algorithm (native units), standard solid, extended dashed
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharey=True)
    for ax, algo in zip(axes.flat, ALGOS):
        _axes_style(ax)
        study = ALGOS[algo][0]
        budget = config(study)['budget']
        versions = [(f'{LABEL[algo]} ({budget:,})', runs(study), budget, False)]
        if algo in EXTENDED and runs(study, EXTENDED[algo]):
            # drawn first, so the standard run stays visible on top of it
            versions.insert(0, (f'{LABEL[algo]} extended ({EXTENDED[algo]:,})', runs(study, EXTENDED[algo]),
                                EXTENDED[algo], True))
        for label, dirs, b, dashed in versions:
            if not dirs:
                continue
            x, m, lo, hi, t = mean_curve(dirs, b, _standard_anchor(study) if dashed else None)
            _series(ax, x, m, lo, hi, algo, f'extended ({b:,})' if dashed else f'standard ({b:,})', dashed,
                    end_label=False)
            rows += [['limit', label, int(xi), '' if t is None else f'{ti:.3f}', f'{mi:.1f}', f'{li:.1f}',
                      f'{hi_:.1f}', len(dirs)]
                     for xi, ti, mi, li, hi_ in zip(x, t if t is not None else x, m, lo, hi)]
        ax.set_title(LABEL[algo], fontsize=10, color=INK, loc='left')
        ax.legend(fontsize=7, frameon=False, loc='lower right')
        ax.set_xlabel(f'training {UNIT[algo]}', fontsize=8)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 3))
    for ax in axes[:, 0]:
        ax.set_ylabel('panel min (mbb/hand, higher = better)', fontsize=8)
    fig.suptitle("Limit hold'em: training-time panel min of the winning configuration "
                 '(mean over 3 seeds, band = range)', fontsize=10, color=INK, x=0.02, ha='left')
    fig.tight_layout()
    _save(fig, 'limit_curves', rows, header)
    plt.close(fig)

    # ---- limit hold'em against wall time: every series on one axis
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _axes_style(ax)
    for algo in ALGOS:
        study = ALGOS[algo][0]
        budget = config(study)['budget']
        for dirs, b, dashed, label in ((runs(study), budget, False, LABEL[algo]),
                                       (runs(study, EXTENDED[algo]) if algo in EXTENDED else [],
                                        EXTENDED.get(algo), True, f'{LABEL[algo]} ext.')):
            if dirs:
                x, m, lo, hi, t = mean_curve(dirs, b, _standard_anchor(study) if dashed else None)
                if t is not None:
                    _series(ax, t, m, lo, hi, algo, label, dashed)
    ax.set_xscale('log')
    _plain_log_axis(ax.xaxis)
    ax.set_xlabel('training wall time (hours, log scale; 3-4 CPU threads per run)', fontsize=8)
    ax.set_ylabel('panel min (mbb/hand, higher = better)', fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    ax.set_title("Limit hold'em: panel min against training time (dashed = extended run)",
                 fontsize=10, color=INK, loc='left')
    fig.tight_layout()
    _save(fig, 'limit_curves_hours', rows, header)
    plt.close(fig)

    # ---- Leduc: exact exploitability, against the budget fraction and against wall time
    from leduc import exploitability, uniform_policy
    uniform = exploitability(uniform_policy) * 1000
    rows = []
    for name, xkey in (('leduc_curves', 'fraction'), ('leduc_curves_hours', 'hours')):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _axes_style(ax)
        for algo in ALGOS:
            study = ALGOS[algo][1]
            if not os.path.isdir(os.path.join(TUNE, study)):
                continue
            budget = config(study)['budget']
            dirs = runs(study)
            if not dirs:
                continue
            x, m, lo, hi, t = mean_curve(dirs, budget)
            xs = x / budget if xkey == 'fraction' else t
            if xs is None:
                continue
            _series(ax, xs, -m, -hi, -lo, algo, LABEL[algo])
            if xkey == 'fraction':
                rows += [['leduc', LABEL[algo], int(xi), '' if t is None else f'{ti:.3f}', f'{-mi:.1f}',
                          f'{-hi_:.1f}', f'{-li:.1f}', len(dirs)]
                         for xi, ti, mi, li, hi_ in zip(x, t if t is not None else x, m, lo, hi)]
        ax.axhline(uniform, color=MUTED, linewidth=1, linestyle=':')
        ax.annotate(f'uniform random ({uniform:.0f})', (0, uniform), xycoords=('axes fraction', 'data'),
                    xytext=(4, 4), textcoords='offset points', fontsize=7, color=INK2)
        ax.set_yscale('log')
        _plain_log_axis(ax.yaxis)
        ax.set_ylabel('exploitability (mbb/hand, log scale, lower = better)', fontsize=8)
        if xkey == 'fraction':
            ax.set_xlabel('fraction of the training budget (DQN / PPO 1M, NFSP 2M episodes; '
                          'Deep CFR 300 iterations)', fontsize=8)
        else:
            ax.set_xscale('log')
            _plain_log_axis(ax.xaxis)
            ax.set_xlabel('training wall time (hours, log scale)', fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4,
                  borderaxespad=0.2)
        ax.set_title("Leduc hold'em: exact exploitability of the winning configuration "
                     '(mean over 3 seeds, band = range)', fontsize=10, color=INK, loc='left', pad=22)
        fig.tight_layout()
        _save(fig, name, rows, header)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Training cost
# --------------------------------------------------------------------------- #

PLAYING = {'dqn': 'q_estimator/qnet', 'ppo': 'network', 'nfsp': 'policy_network', 'deepcfr': 'policy_network'}


def parameters(algo, d):
    ''' (parameters of the network that plays, of every trained network);
    optimizer states and target networks are not counted. '''
    import torch
    obj = torch.load(os.path.join(d, ALGOS[algo][2]), map_location='cpu', weights_only=False)
    counts = {}

    def walk(o, path):
        if isinstance(o, torch.Tensor):
            counts[path] = o.numel()
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f'{path}/{k}' if path else str(k))
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f'{path}/{i}')
    walk(obj, '')
    counts = {k: v for k, v in counts.items() if 'optim' not in k.lower() and 'target' not in k.lower()}
    playing = sum(v for k, v in counts.items() if k.startswith(PLAYING[algo] + '/') or k == PLAYING[algo])
    return playing, sum(counts.values())


def tuning_effort(study):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    import tune
    # load_study, not tune.load_study (create_study appends a 'study created' entry to the journal)
    s = optuna.load_study(study_name=study, storage=tune.storage(study))
    states, seconds = {}, 0.0
    for t in s.get_trials(deepcopy=False):
        states[t.state.name] = states.get(t.state.name, 0) + 1
        if t.duration:
            seconds += t.duration.total_seconds()
    return states, seconds / 3600


def _agg(xs, digits=1):
    xs = [x for x in xs if x is not None]
    if not xs:
        return ''
    return f'{statistics.fmean(xs):.{digits}f}' + (f' (sd {statistics.stdev(xs):.{digits}f})' if len(xs) > 1 else '')


def cmd_cost(a):
    from final_evaluation import _write_table
    header = ['game', 'algorithm', 'study', 'budget', 'runs', 'hours per run', 'parameters (playing)',
              'parameters (all trained)', 'tuning trials (complete / pruned / failed)', 'tuning worker-hours']
    rows = []
    for game, col in (('limit', 0), ('leduc', 1)):
        for algo in ALGOS:
            study = ALGOS[algo][col]
            if not os.path.isdir(os.path.join(TUNE, study)):
                continue
            states, hours = tuning_effort(study)
            trials = f'{states.get("COMPLETE", 0)} / {states.get("PRUNED", 0)} / {states.get("FAIL", 0)}'
            versions = [(LABEL[algo], runs(study), config(study)['budget'], trials, f'{hours:.0f}')]
            if game == 'limit' and algo in EXTENDED and runs(study, EXTENDED[algo]):
                versions.append((LABEL[algo] + ' extended', runs(study, EXTENDED[algo]), EXTENDED[algo], '-', '-'))
            for label, dirs, budget, trials_txt, hours_txt in versions:
                p = parameters(algo, dirs[0]) if dirs else (None, None)
                rows.append([game, label, study, f'{budget:,} {UNIT[algo]}', len(dirs),
                             _agg([wall_hours(d) for d in dirs]),
                             f'{p[0]:,}' if p[0] else '', f'{p[1]:,}' if p[1] else '', trials_txt, hours_txt])
    md = _write_table(os.path.join(OUT, 'cost'), 'Training cost of the winning configurations', header, rows,
                      'Hours per run: wall time of one training run (3-4 CPU threads, 4 runs in parallel on a '
                      'laptop), mean and sd over seeds. Parameters: the network that plays (DQN: Q-network; '
                      'PPO: actor-critic network; NFSP: average-policy network; Deep CFR: policy network) and '
                      'every network trained (NFSP adds its best-response Q-network, Deep CFR its two '
                      'advantage networks); optimizer states and target networks are not counted. Tuning '
                      'worker-hours: summed wall time of all trials of the study.')
    with open(os.path.join(OUT, 'cost.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    print(md)


# --------------------------------------------------------------------------- #
# Leduc: exact round robin
# --------------------------------------------------------------------------- #

def cmd_leduc_h2h(a):
    import torch
    from agent_loading import load_policy
    from leduc import match_value, policy_of, equilibrium, exploitability
    from final_evaluation import _write_table, _ranks
    torch.set_num_threads(a.threads)
    agents = []                       # (algo, name, probs function, exploitability)
    for algo in ALGOS:
        study = ALGOS[algo][1]
        if not os.path.isdir(os.path.join(TUNE, study)):
            continue
        with open(os.path.join(TUNE, study, 'finalists', 'finalists.json'), encoding='utf-8') as f:
            record = json.load(f)
        w = record['winner']
        for d in runs(study):
            seed = int(d.rsplit('seed_', 1)[1])
            run = next(r for r in record['runs'] if r['trial'] == w['trial'] and r['seed'] == seed)
            agents.append((algo, f'{algo}_s{seed}', policy_of(load_policy(algo, d)), -run['final_min']))
    eq = equilibrium()
    start = time.perf_counter()
    value = {}
    for i, (x_algo, x, px, _) in enumerate(agents):
        value[x, 'cfr+'] = match_value(px, eq, b_by_key=True) * 1000
        for y_algo, y, py, _ in agents[i + 1:]:
            if y_algo != x_algo:
                value[x, y] = match_value(px, py) * 1000
                value[y, x] = -value[x, y]
    print(f'{len(value)} exact matches in {time.perf_counter() - start:.0f}s')
    algos = list(dict.fromkeys(a_ for a_, _, _, _ in agents))
    md = ['# Leduc hold\'em: exact head-to-head\n',
          'Every winner seed against every seed of the other algorithms, computed exactly over all deals '
          '(no card noise). mbb/hand from the row\'s view, averaged over both seats.\n']

    names = [n for _, n, _, _ in agents]
    rows = [[x] + ['--' if (x, y) not in value else f'{value[x, y]:+.0f}' for y in names + ['cfr+']]
            for x in names]
    md.append(_write_table(os.path.join(OUT, 'leduc_h2h_agents'), 'Leduc, exact head-to-head, every agent',
                           ['agent'] + names + ['CFR+'], rows))

    def cell(xa, ya):
        vals = [value[x, y] for xa_, x, _, _ in agents for ya_, y, _, _ in agents
                if xa_ == xa and ya_ == ya and (x, y) in value]
        return vals
    rows = []
    for xa in algos:
        row = [LABEL[xa]]
        for ya in algos:
            v = cell(xa, ya) if xa != ya else []
            row.append('--' if xa == ya else f'{statistics.fmean(v):+.0f} (sd {statistics.stdev(v):.0f})')
        v = [value[x, 'cfr+'] for xa_, x, _, _ in agents if xa_ == xa]
        row.append(f'{statistics.fmean(v):+.0f} (sd {statistics.stdev(v):.0f})' if len(v) > 1 else f'{v[0]:+.0f}')
        rows.append(row)
    md.append(_write_table(os.path.join(OUT, 'leduc_h2h_algorithms'),
                           'Leduc, exact head-to-head by algorithm (mean over seed pairs, sd = spread)',
                           ['algorithm'] + [LABEL[x] for x in algos] + ['CFR+ equilibrium'], rows))

    # ranking by head-to-head against ranking by exploitability
    per_agent = []
    for xa, x, _, expl in agents:
        others = [value[x, y] for ya, y, _, _ in agents if ya != xa]
        per_agent.append((xa, x, statistics.fmean(others), expl))
    rows = []
    for xa in algos:
        mine = [p for p in per_agent if p[0] == xa]
        rows.append([LABEL[xa], f'{statistics.fmean(p[2] for p in mine):+.0f}',
                     f'{statistics.fmean(p[3] for p in mine):.0f}'])
    rows.sort(key=lambda r: -float(r[1]))
    h = _ranks([p[2] for p in per_agent])
    e = _ranks([-p[3] for p in per_agent])
    rho = float(np.corrcoef(h, e)[0, 1])
    md.append(_write_table(os.path.join(OUT, 'leduc_h2h_ranking'),
                           'Leduc: ranking by head-to-head and by exploitability',
                           ['algorithm', 'mean head-to-head vs others (mbb/hand)', 'exploitability (mbb/hand)'],
                           rows, f'Spearman rank correlation over the {len(per_agent)} agents between the mean '
                                 f'head-to-head result and exploitability (negated, so +1 = same order): '
                                 f'{rho:+.2f}.'))
    with open(os.path.join(OUT, 'leduc_h2h.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print('\n'.join(md))


# --------------------------------------------------------------------------- #
# Tuning: winners, search spaces, parameter importance
# --------------------------------------------------------------------------- #

def _load_study_readonly(study):
    import optuna
    import tune
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # optuna.load_study, not tune.load_study (create_study appends a 'study created' entry to the journal)
    return optuna.load_study(study_name=study, storage=tune.storage(study))


def _range_text(spec):
    if not spec.get('tune', True):
        return f'fixed {spec["value"]}'
    if spec['type'] == 'categorical':
        return ' / '.join(str(c) for c in spec['choices'])
    return f'{spec["low"]:g} - {spec["high"]:g}' + (' (log)' if spec.get('log') else '')


def _value_text(v):
    return f'{v:.3g}' if isinstance(v, float) else str(v)


def _importances(trials):
    ''' fANOVA importances of the trials' parameters for their values
    (None if there are too few trials). '''
    import optuna
    if len(trials) < 5:
        return None
    s = optuna.create_study(direction='maximize')
    s.add_trials(trials)
    try:
        return optuna.importance.get_param_importances(s, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
    except Exception:
        return None


def cmd_tuning(a):
    ''' Per study: the search space, the winner's values and the parameter
    importances, from the complete trials and (because successive halving
    prunes most trials) from every trial's score at the first checkpoint at
    which pruning could start, which all trials that did not fail reached. '''
    import optuna
    from optuna.trial import TrialState
    from final_evaluation import _write_table
    out = os.path.join(OUT, 'tuning')
    os.makedirs(out, exist_ok=True)
    md = ['# Hyper-parameter tuning\n']
    for game, col in (('limit', 0), ('leduc', 1)):
        for algo in ALGOS:
            study = ALGOS[algo][col]
            if not os.path.isdir(os.path.join(TUNE, study)):
                continue
            cfg = config(study)
            s = _load_study_readonly(study)
            trials = s.get_trials(deepcopy=False)
            w = winner(study)
            complete = [t for t in trials if t.state == TrialState.COMPLETE]
            step = max(1, round(cfg['min_resource_frac'] * cfg['evals'])) - 1 if cfg['prune'] else cfg['evals'] - 1
            early = [optuna.trial.create_trial(params=t.params, distributions=t.distributions,
                                               value=t.intermediate_values[step])
                     for t in trials if step in t.intermediate_values and t.params]
            imp_complete, imp_early = _importances(complete), _importances(early)
            states = {}
            for t in trials:
                states[t.state.name] = states.get(t.state.name, 0) + 1
            rows = []
            for name, spec in cfg['space'].items():
                tuned = spec.get('tune', True)
                rows.append([name, _range_text(spec), _value_text(w['params'][name]) if tuned else '(fixed)',
                             f'{imp_complete[name]:.2f}' if tuned and imp_complete and name in imp_complete else '',
                             f'{imp_early[name]:.2f}' if tuned and imp_early and name in imp_early else ''])
            fixed = ', '.join(f'{k}={v}' for k, v in cfg['fixed'].items() if k != 'verbose')
            title = f'{LABEL[algo]}, {game}: study {study}'
            md.append(_write_table(
                os.path.join(out, study), title,
                ['parameter', 'searched', 'winner', f'importance ({len(complete)} complete)',
                 f'importance ({len(early)} trials at evaluation {step + 1})'], rows,
                f'{len(trials)} trials ({", ".join(f"{k.lower()} {v}" for k, v in sorted(states.items()))}); '
                f'budget {cfg["budget"]:,} {UNIT[algo]}, {cfg["evals"]} evaluations per trial, successive-halving '
                f'pruning; winner = trial {w["trial"]}, chosen by `tune.py finalists` (top 3 trials x 3 new '
                f'seeds). Fixed settings: {fixed}. Importance: fANOVA share of the score variance explained by '
                f'each parameter, over the complete trials, and over every trial\'s score at evaluation '
                f'{step + 1}, the first one at which pruning could stop a trial (so all trials reached it); '
                f'blank = too few trials.'))
            _optimization_history(trials, os.path.join(out, f'{study}_history'), title)
    with open(os.path.join(out, 'tuning.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f'  tuning/: {len(md) - 1} studies (tuning.md, <study>.csv/.tex, <study>_history.png/.pdf)')


def _optimization_history(trials, path, title):
    ''' Every trial's last reported score (complete = filled, pruned = hollow)
    and the running best of the complete trials. '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from optuna.trial import TrialState
    fig, ax = plt.subplots(figsize=(7, 3.6))
    _axes_style(ax)
    best, xs, ys = -np.inf, [], []
    for t in sorted(trials, key=lambda t: t.number):
        if not t.intermediate_values:
            continue
        v = t.value if t.state == TrialState.COMPLETE else t.intermediate_values[max(t.intermediate_values)]
        done = t.state == TrialState.COMPLETE
        ax.plot(t.number, v, 'o', markersize=5, color='#2a78d6' if done else MUTED,
                markerfacecolor='#2a78d6' if done else 'none')
        if done:
            best = max(best, v)
            xs.append(t.number)
            ys.append(best)
    if xs:
        ax.step(xs, ys, where='post', color=INK2, linewidth=1.5, label='best complete trial so far')
    ax.plot([], [], 'o', color='#2a78d6', label='complete (final score)')
    ax.plot([], [], 'o', color=MUTED, markerfacecolor='none', label='pruned (last score)')
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    ax.set_xlabel('trial', fontsize=8)
    ax.set_ylabel('tuning score (mbb/hand)', fontsize=8)
    ax.set_title(title, fontsize=9, color=INK, loc='left')
    fig.tight_layout()
    fig.savefig(path + '.png', dpi=200, facecolor=SURFACE)
    fig.savefig(path + '.pdf', facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# NFSP: the diverged studies nfsp_v1 / nfsp_v2 against nfsp_v3
# --------------------------------------------------------------------------- #

def _q_network_finite(d):
    ''' False if the best-response Q-network of an NFSP checkpoint has a
    non-finite weight. '''
    import torch
    obj = torch.load(os.path.join(d, 'checkpoint_nfsp.pt'), map_location='cpu', weights_only=False)
    bad = []

    def walk(o, path):
        if isinstance(o, torch.Tensor):
            if o.is_floating_point() and not torch.isfinite(o).all():
                bad.append(path)
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f'{path}/{k}')
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f'{path}/{i}')
    walk(obj.get('rl_agent', obj), '')
    return not [p for p in bad if 'optim' not in p.lower()]


def cmd_nfsp_divergence(a):
    from optuna.trial import TrialState
    from final_evaluation import _write_table
    rows = []
    for study in ('nfsp_v1', 'nfsp_v2', 'nfsp_v3'):
        if not os.path.isdir(os.path.join(TUNE, study)):
            continue
        cfg = config(study)
        space = cfg['space']
        trials = _load_study_readonly(study).get_trials(deepcopy=False)
        states = {}
        for t in trials:
            states[t.state.name] = states.get(t.state.name, 0) + 1
        complete = [t for t in trials if t.state == TrialState.COMPLETE]
        best = max(complete, key=lambda t: t.value) if complete else None
        row = [study, space['optimizer'].get('value', 'tuned'), _range_text(space['rl_learning_rate']),
               _range_text(space['max_grad_norm']) if 'max_grad_norm' in space else 'none',
               f'{len(trials)} ({states.get("COMPLETE", 0)} / {states.get("PRUNED", 0)} / {states.get("FAIL", 0)})',
               f'{best.value:+.0f} (trial {best.number}, rl lr {best.params["rl_learning_rate"]:.3g})' if best else '']
        fin = os.path.join(TUNE, study, 'finalists', 'finalists.json')
        if os.path.isfile(fin):
            with open(fin, encoding='utf-8') as f:
                record = json.load(f)
            dirs = [os.path.join(REPO_ROOT, r['log_dir']) for r in record['runs'] if 'failed' not in r]
            dirs = [d for d in dirs if os.path.isfile(os.path.join(d, 'checkpoint_nfsp.pt'))]
            nan = sum(not _q_network_finite(d) for d in dirs)
            w = record.get('winner')
            held = []
            for r in record['runs']:
                p = os.path.join(REPO_ROOT, r['log_dir'], 'heldout.json')
                if w and r['trial'] == w['trial'] and os.path.isfile(p):
                    with open(p, encoding='utf-8') as f:
                        held.append(json.load(f))
            fold = [h['play_vs_tag']['fold facing a bet']['value'] for h in held
                    if h['play_vs_tag'].get('fold facing a bet', {}).get('value') is not None]
            row += [f'{nan} / {len(dirs)}',
                    _agg([h['panel']['min'] for h in held], 0), _agg([h['lbr']['mean'] for h in held], 0),
                    f'{100 * statistics.fmean(fold):.0f}%' if fold else '']
        else:
            row += ['no finalists', '', '', '']
        rows.append(row)
    md = _write_table(os.path.join(OUT, 'nfsp_divergence'),
                      'NFSP tuning studies: diverged best-response Q-networks (v1, v2) and the fixed study (v3)',
                      ['study', 'optimizer', 'rl learning rate searched', 'gradient clipping',
                       'trials (complete / pruned / failed)', 'best tuning score',
                       'finalists with a non-finite Q-network', 'held-out panel min (winner)', 'LBR (winner)',
                       'fold facing a bet vs tag (winner)'], rows,
                      'In nfsp_v1 / nfsp_v2 (plain SGD, large RL learning rates, no divergence check) the '
                      'best-response Q-network diverged to NaN; the greedy action then silently became the first '
                      'legal one (call, else raise), so the average policy learned never to fold, which the '
                      'panel rewarded and the tuner selected. nfsp_v3 uses Adam, smaller learning rates, gradient '
                      'clipping and a divergence check that fails such trials. Held-out values: mean (sd) over '
                      'the winner\'s seeds; fold rate from the held-out match against tag.')
    with open(os.path.join(OUT, 'nfsp_divergence.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    print(md)


# --------------------------------------------------------------------------- #
# Example hands
# --------------------------------------------------------------------------- #

HAND_FILTERS = [('river_bluff', 2), ('bluff', 2), ('slowplay', 2), ('big_win', 2), ('big_loss', 2)]


def cmd_hands(a):
    ''' Each winner (first seed) plays tag; characteristic hands are written
    as text replays (for the thesis), HTML and all hands as JSON lines. '''
    import torch
    import behaviour_tracker as bt
    from agent_loading import load_policy
    torch.set_num_threads(a.threads)
    out = os.path.join(OUT, 'hands')
    os.makedirs(out, exist_ok=True)
    tag = load_policy('tag')
    agents = [(algo, LABEL[algo], runs(ALGOS[algo][0])[0]) for algo in ALGOS]
    agents += [(f'{algo}_ext', LABEL[algo] + ' ext.', runs(ALGOS[algo][0], EXTENDED[algo])[0])
               for algo in EXTENDED if runs(ALGOS[algo][0], EXTENDED[algo])]
    jobs = []
    for name, label, d in agents:
        kind = name.replace('_ext', '')
        hands = bt.play_match(load_policy(kind, d), tag, a.hand_deals, names=(label, 'TAG'))
        bt.save_hands(hands, os.path.join(out, f'{name}_vs_tag.jsonl'))
        text, chosen = [], []
        for filt, n in HAND_FILTERS:
            picked = [h for h in bt.select_hands(hands, filt, role='agent', count=n + len(chosen))
                      if h not in chosen][:n]
            chosen += picked
            for h in picked:
                h['category'] = filt
            text.append(f'===== {filt} ({len(picked)} of {sum(bt.FILTERS[filt](h, "agent") for h in hands)} '
                        f'matching hands) =====')
            text += [bt.render_hand(h) + '\n' for h in picked]
        with open(os.path.join(out, f'{name}_vs_tag.txt'), 'w', encoding='utf-8') as f:
            f.write(f'{label} ({os.path.relpath(d, REPO_ROOT)}) vs TAG, {a.hand_deals} duplicate deals\n\n'
                    + '\n'.join(text))
        with open(os.path.join(out, f'{name}_vs_tag.html'), 'w', encoding='utf-8') as f:
            f.write(bt.render_html(chosen, title=f'{label} vs TAG: example hands'))
        with open(os.path.join(out, f'{name}_vs_tag.selected.json'), 'w', encoding='utf-8') as f:
            json.dump([{'deal': h['deal'], 'swap': h['swap'], 'category': h['category']} for h in chosen], f)
        print(f'  hands/{name}_vs_tag.txt / .html / .jsonl ({len(chosen)} hands selected of {len(hands)})')
        jobs.append([sys.executable, '-u', 'hand_analysis.py', 'limit', '--hands', f'{name}_vs_tag.jsonl',
                     '--select', f'{name}_vs_tag.selected.json', '--kind', kind, '--ckpt', d,
                     '--opponent', 'tag', '--samples', str(a.hand_samples),
                     '--out', f'{name}_vs_tag.analysis.json'])
    # Leduc: each winner (first seed) against the CFR+ equilibrium, exact
    for algo in ALGOS:
        study = ALGOS[algo][1]
        if os.path.isdir(os.path.join(TUNE, study)) and runs(study):
            jobs.append([sys.executable, '-u', 'hand_analysis.py', 'leduc', '--kind', algo, '--ckpt', runs(study)[0],
                         '--name', LABEL[algo], '--out', f'leduc_{algo}_vs_cfr.analysis.json'])
    # the mistake measure (hand_analysis.py), 4 processes at a time
    running = []
    for job in jobs:
        job = [os.path.join(out, x) if x.endswith(('.jsonl', '.json')) else x for x in job]
        while len(running) >= a.processes:
            running = [p for p in running if p.poll() is None]
            time.sleep(2)
        log = open(job[-1].replace('.analysis.json', '.analysis.log'), 'w', encoding='utf-8')
        running.append(subprocess.Popen(job, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT))
    for p in running:
        p.wait()
    failed = [j for j in jobs if not os.path.isfile(os.path.join(out, j[-1]))]
    print(f'  mistake measure: {len(jobs) - len(failed)} of {len(jobs)} analyses written'
          + (f'; FAILED: {[j[-1] for j in failed]} (see the .analysis.log files)' if failed else ''))
    subprocess.call([sys.executable, '-u', os.path.join(REPO_ROOT, 'hand_viewer.py'), '--dir', out,
                     '--out', os.path.join(out, 'hands.html')], cwd=REPO_ROOT)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

def _git(*args):
    try:
        return subprocess.run(['git', *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def cmd_provenance(a):
    from importlib import metadata
    versions = {}
    for pkg in ('torch', 'numpy', 'rlcard', 'optuna', 'matplotlib', 'scipy'):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    status = (_git('status', '--porcelain') or '').splitlines()
    code = [line[3:] for line in status if line[3:].endswith('.py')]
    record = {
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'git_commit': (_git('rev-parse', 'HEAD') or '').strip() or None,
        'git_branch': (_git('rev-parse', '--abbrev-ref', 'HEAD') or '').strip() or None,
        'uncommitted_python_files': code,
        'uncommitted_files_total': len(status),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'packages': versions,
        'command': ' '.join(sys.argv),
    }
    for name in ('manifest.json', 'results.json'):
        path = os.path.join(OUT, name)
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                record[name.replace('.json', '_sha256')] = hashlib.sha256(f.read()).hexdigest()
    results = os.path.join(OUT, 'results.json')
    if os.path.isfile(results):
        with open(results, encoding='utf-8') as f:
            record['final_evaluation_settings'] = json.load(f)['settings']
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, a.name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=1)
    print(f'{path}: commit {record["git_commit"]}, {len(code)} uncommitted Python file(s)')
    if code:
        print('  WARNING: uncommitted code changes -- the commit alone does not reproduce these results: '
              + ', '.join(code))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('curves')
    sub.add_parser('cost')
    h = sub.add_parser('leduc_h2h')
    sub.add_parser('tuning')
    sub.add_parser('nfsp_divergence')
    hd = sub.add_parser('hands')
    pr = sub.add_parser('provenance')
    pr.add_argument('--name', default='provenance.json')
    al = sub.add_parser('all')
    for q in (h, hd, al):
        q.add_argument('--threads', type=int, default=4)
    for q in (hd, al):
        q.add_argument('--hand_deals', type=int, default=300, help='duplicate deals per agent against tag')
        q.add_argument('--hand_samples', type=int, default=300,
                       help='opponent holdings drawn per decision for the mistake measure')
        q.add_argument('--processes', type=int, default=4, help='hand analyses run at once')
    al.add_argument('--name', default='provenance.json')
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    commands = {'curves': [cmd_curves], 'cost': [cmd_cost], 'leduc_h2h': [cmd_leduc_h2h],
                'tuning': [cmd_tuning], 'nfsp_divergence': [cmd_nfsp_divergence], 'hands': [cmd_hands],
                'provenance': [cmd_provenance],
                'all': [cmd_provenance, cmd_cost, cmd_leduc_h2h, cmd_curves, cmd_tuning, cmd_nfsp_divergence,
                        cmd_hands]}[a.command]
    for cmd in commands:
        cmd(a)


if __name__ == '__main__':
    main()
