''' Collects the thesis tables and figures in one folder, experiments/thesis/:

  tables/   LaTeX (booktabs) tables, each also as .csv
            - copies of the final-evaluation, Leduc, tuning, NFSP-divergence and
              ablation tables (algorithm keys such as ppo\\_ext relabelled "PPO ext.")
            - overview        one row per algorithm: budget, wall time, head-to-head,
                              held-out panel min, LBR
            - ablations       the four ablations side by side (paired differences
                              and head-to-head against the original)
            - leduc_mistakes  EV loss per decision against CFR+ over 400 deals,
                              next to exploitability and head-to-head
  figures/  PDF (for LaTeX) and PNG
            - copies of the learning curves and tuning histories
            - h2h_matrix, leduc_h2h_matrix  round robin by algorithm (heatmap)
            - h2h_ranking     mean head-to-head per algorithm with its six matchups
            - measures        per agent: head-to-head against LBR
            - ablations       head-to-head of each changed arm against the original,
                              and the LBR change per seed
  README.md index: every file with its caption and source

Reads only finished results (experiments/final/, experiments/ablation_*.json);
trains and evaluates nothing.

    python thesis_assets.py
'''

import os
import re
import csv
import json
import glob
import shutil
import statistics

import numpy as np

from final_report import (COLOR, MARKER, LABEL, INK, INK2, MUTED, GRID, SURFACE, _axes_style)
from final_evaluation import _write_table, _h2h_algo_se

REPO = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(REPO, 'experiments', 'final')
EXP = os.path.join(REPO, 'experiments')
OUT = os.path.join(EXP, 'thesis')
TABLES = os.path.join(OUT, 'tables')
FIGURES = os.path.join(OUT, 'figures')

# diverging scale of the dataviz reference palette: red (row loses) - neutral gray - blue (row wins)
DIV_NEG, DIV_MID, DIV_POS = '#e34948', '#f0efec', '#2a78d6'

ORDER = ['dqn', 'ppo', 'ppo_ext', 'nfsp', 'nfsp_ext', 'deepcfr', 'deepcfr_ext']
COST_NAME = {'DQN': 'dqn', 'PPO': 'ppo', 'PPO extended': 'ppo_ext', 'NFSP': 'nfsp', 'NFSP extended': 'nfsp_ext',
             'Deep CFR': 'deepcfr', 'Deep CFR extended': 'deepcfr_ext'}

ABLATIONS = [  # (name, algorithm, what changed, arms compared with A)
    ('ppo_free_fold', 'ppo', 'free fold removed', [('B', 'at evaluation'), ('C', 'retrained')]),
    ('nfsp_batchnorm', 'nfsp', 'BatchNorm removed', [('C', 'retrained')]),
    ('deepcfr_sgd', 'deepcfr', '4000 SGD steps per fit (was 1000)', [('C', 'retrained')]),
    ('nfsp_literature', 'nfsp', 'literature hyperparameters (8M episodes)', [('C', 'retrained')]),
]

INDEX = []   # (section, file, caption, source) for README.md


def base(key):
    return key.split('_ext')[0]


def label(key):
    return LABEL[base(key)] + (' ext.' if key.endswith('_ext') else '')


def note(section, file, caption, source):
    INDEX.append((section, file, caption, source))


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.reader(f))


def pm(text):
    ''' "-83 ± 52 (sd 80)" -> (-83.0, 52.0); "+146 (sd 225)" -> (146.0, None). '''
    m = re.match(r'\s*([+-]?\d+(?:\.\d+)?)(?:\s*±\s*(\d+(?:\.\d+)?))?', text)
    return (float(m.group(1)), float(m.group(2)) if m.group(2) else None) if m else (None, None)


def agg(xs, digits=0):
    s = f'{statistics.fmean(xs):+.{digits}f}'
    return s + (f' (sd {statistics.stdev(xs):.{digits}f})' if len(xs) > 1 else '')


# --------------------------------------------------------------------------- #
# Copies of existing tables and figures
# --------------------------------------------------------------------------- #

_KEY = re.compile(r'\b(dqn|ppo|nfsp|deepcfr)((?:\\_ext)?)((?:\\_s\d+)?)(?![\w\\])')


def relabel_tex(text):
    ''' ppo\\_ext -> PPO ext., deepcfr\\_s1042 -> Deep CFR (1042); study names (nfsp\\_v3) are left alone. '''
    def sub(m):
        seed = m.group(3).replace('\\_s', '')
        return LABEL[m.group(1)] + (' ext.' if m.group(2) else '') + (f' ({seed})' if seed else '')
    return '\n'.join(l if l.startswith('%') else _KEY.sub(sub, l) for l in text.split('\n'))


COPY_TABLES = [  # (source relative to experiments/, target name, relabel, caption)
    ('final/h2h_ranking', 'h2h_ranking', True, "Limit hold'em: ranking by the head-to-head round robin"),
    ('final/h2h_algorithms', 'h2h_algorithms', True, "Limit hold'em: head-to-head by algorithm (mean ± se, sd = seed spread)"),
    ('final/h2h_agents', 'h2h_agents', True, "Limit hold'em: head-to-head of every agent (appendix)"),
    ('final/algorithms', 'algorithms', True, "Limit hold'em: held-out panel and LBR by algorithm"),
    ('final/agents', 'agents', True, "Limit hold'em: held-out panel and LBR of every agent (appendix)"),
    ('final/measure_correlation', 'measure_correlation', False, 'Spearman correlation between the measures'),
    ('final/h2h_style', 'h2h_style', True, 'Playing style in the round robin'),
    ('final/h2h_style_detail', 'h2h_style_detail', True, 'Playing style in detail (appendix)'),
    ('final/h2h_endings', 'h2h_endings', True, 'How the head-to-head hands end (showdown / fold) and where the money is won'),
    ('final/h2h_position', 'h2h_position', True, 'Head-to-head by position (small blind / big blind)'),
    ('final/cost', 'cost', False, 'Training cost: budget, wall time, network size, tuning effort'),
    ('final/leduc_h2h_algorithms', 'leduc_h2h_algorithms', False, 'Leduc: exact head-to-head by algorithm, and against CFR+'),
    ('final/leduc_h2h_agents', 'leduc_h2h_agents', False, 'Leduc: exact head-to-head of every agent (appendix)'),
    ('final/leduc_h2h_ranking', 'leduc_h2h_ranking', False, 'Leduc: head-to-head ranking against exploitability'),
    ('final/nfsp_divergence', 'nfsp_divergence', False, 'The diverged NFSP studies v1 / v2 against the fixed v3'),
]
for _n in ('ppo_free_fold', 'nfsp_batchnorm', 'deepcfr_sgd', 'nfsp_literature'):
    for _suffix, _what in (('', 'all measures per arm'), ('_diff', 'paired differences per seed'),
                           ('_h2h', 'head-to-head'), ('_training', 'training-time score and hours')):
        COPY_TABLES.append((f'ablation_{_n}{_suffix}', f'ablation_{_n}{_suffix}', True, f'Ablation {_n}: {_what}'))


def copy_existing():
    for src, dst, relabel, caption in COPY_TABLES:
        for ext in ('.tex', '.csv'):
            p = os.path.join(EXP, *src.split('/')) + ext
            if not os.path.isfile(p):
                print(f'  missing {p}')
                continue
            with open(p, encoding='utf-8') as f:
                text = f.read()
            if relabel and ext == '.tex':
                text = relabel_tex(text)
            with open(os.path.join(TABLES, dst + ext), 'w', encoding='utf-8', newline='') as f:
                f.write(text)
        note('Tables', f'tables/{dst}.tex', caption, f'experiments/{src}.tex')
    for p in sorted(glob.glob(os.path.join(FINAL, 'tuning', '*.tex'))):
        n = os.path.basename(p)[:-4]
        for ext in ('.tex', '.csv'):
            if os.path.isfile(p[:-4] + ext):
                shutil.copyfile(p[:-4] + ext, os.path.join(TABLES, f'tuning_{n}{ext}'))
        note('Tables', f'tables/tuning_{n}.tex', f'Tuning of {n}: search space, winner, parameter importance',
             f'experiments/final/tuning/{n}.tex')

    figures = [('limit_curves', "Limit hold'em learning curves (training-time panel min, mean over 3 seeds, band = range)"),
               ('limit_curves_hours', "Limit hold'em panel min against training wall time"),
               ('leduc_curves', 'Leduc learning curves (exact exploitability)'),
               ('leduc_curves_hours', 'Leduc exploitability against training wall time')]
    for n, caption in figures:
        for ext in ('.pdf', '.png'):
            p = os.path.join(FINAL, 'figures', n + ext)
            if os.path.isfile(p):
                shutil.copyfile(p, os.path.join(FIGURES, n + ext))
        note('Figures', f'figures/{n}.pdf', caption, f'experiments/final/figures/{n}')
    for p in sorted(glob.glob(os.path.join(FINAL, 'tuning', '*_history.png'))):
        n = os.path.basename(p)[:-4]
        for ext in ('.pdf', '.png'):
            if os.path.isfile(p[:-4] + ext):
                shutil.copyfile(p[:-4] + ext, os.path.join(FIGURES, f'tuning_{n}{ext}'))
        note('Figures', f'figures/tuning_{n}.png', f'Optimization history of {n[:-8]}', f'experiments/final/tuning/{n}')


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def limit_cells():
    ''' {(row, col): (mean, se)} by algorithm, from h2h_algorithms.csv. '''
    rows = read_csv(os.path.join(FINAL, 'h2h_algorithms.csv'))
    cols = rows[0][1:]
    return {(r[0], c): pm(v) for r in rows[1:] for c, v in zip(cols, r[1:]) if v != '--'}


def ranking():
    return [(r[0], float(r[1]), r[2], r[3]) for r in read_csv(os.path.join(FINAL, 'h2h_ranking.csv'))[1:]]


def agent_results():
    with open(os.path.join(FINAL, 'results.json'), encoding='utf-8') as f:
        tasks = json.load(f)['tasks']
    h2h = {}
    for k, v in tasks.items():
        if k.startswith('h2h:'):
            _, a, b = k.split(':')
            h2h[a, b] = v['mean']
            h2h[b, a] = -v['mean']
    agents = sorted({k.split(':')[1] for k in tasks if k.startswith('lbr:')})
    out = {}
    for a in agents:
        algo = a.rsplit('_s', 1)[0]
        others = [h2h[a, b] for b in agents if b.rsplit('_s', 1)[0] != algo and (a, b) in h2h]
        out[a] = {'algo': algo, 'lbr': tasks[f'lbr:{a}']['mean'], 'panel_min': tasks[f'panel:{a}']['min'],
                  'h2h': statistics.fmean(others)}
    return out


def ablation(name):
    with open(os.path.join(EXP, f'ablation_{name}.json'), encoding='utf-8') as f:
        d = json.load(f)
    m, h2h = d['measurements'], d['h2h']
    seeds = sorted({int(k.rsplit('_s', 1)[1]) for k in m if k.startswith('A_s')})
    return m, h2h, seeds


def arm_vs_a(m, h2h, seeds, arm):
    ''' Paired differences arm - A per seed, and the head-to-head of arm against A. '''
    both = [s for s in seeds if f'{arm}_s{s}' in m and f'A_s{s}' in m]
    diff = {k: [f(m[f'{arm}_s{s}']) - f(m[f'A_s{s}']) for s in both]
            for k, f in (('panel_min', lambda r: r['panel']['min']), ('panel_mean', lambda r: r['panel']['mean']),
                         ('lbr', lambda r: r['lbr']['mean']))}
    grid = np.full((len(seeds), len(seeds)), np.nan)
    var = []
    for i, s in enumerate(seeds):
        for j, t in enumerate(seeds):
            r = h2h.get(f'{arm}_s{s}:A_s{t}')
            if r:
                grid[i, j] = r['mean']
                var.append(r['se'] ** 2)
    return diff, (float(np.nanmean(grid)), _h2h_algo_se(grid, var))


# --------------------------------------------------------------------------- #
# New tables
# --------------------------------------------------------------------------- #

def table_overview():
    cost = {COST_NAME[r[1]]: r for r in read_csv(os.path.join(FINAL, 'cost.csv'))[1:] if r[0] == 'limit'}
    algos = {r[0]: r for r in read_csv(os.path.join(FINAL, 'algorithms.csv'))[1:]}
    header = ['algorithm', 'training budget', 'hours per run', 'head-to-head vs others', 'significant wins',
              'held-out panel min', 'LBR']
    rows = []
    for key, mean, wins, sig in ranking():
        c, a = cost[key], algos[key]
        rows.append([label(key), c[3], c[5], f'{mean:+.0f}', sig, a[2], a[9].lstrip('+')])
    _write_table(os.path.join(TABLES, 'overview'),
                 "Limit hold'em: overview of the seven agents (3 seeds each), ordered by the head-to-head round robin",
                 header, rows)
    note('Tables', 'tables/overview.tex',
         "Overview: budget, wall time per run, mean head-to-head against the other six algorithms (mbb/hand), "
         'opponents beaten by more than 2 se, held-out panel min and LBR (mean (sd) over seeds; lower LBR = less exploitable)',
         'experiments/final/h2h_ranking.csv, algorithms.csv, cost.csv')


def table_ablations():
    header = ['algorithm', 'change', 'arm', 'panel min', 'panel mean', 'LBR', 'head-to-head vs original']
    rows = []
    for name, algo, change, arms in ABLATIONS:
        m, h2h, seeds = ablation(name)
        for arm, how in arms:
            diff, (hm, hse) = arm_vs_a(m, h2h, seeds, arm)
            rows.append([LABEL[algo], change, f'{arm} ({how})', agg(diff['panel_min']), agg(diff['panel_mean']),
                         agg(diff['lbr']), f'{hm:+.0f} ± {hse:.0f}'])
    _write_table(os.path.join(TABLES, 'ablations'),
                 'Ablations: changed arm minus the original A, paired by seed (mean (sd) over 3 seeds), '
                 'and the changed arm against A head-to-head (mean ± se, 2000 duplicate deals per seed pair)',
                 header, rows)
    note('Tables', 'tables/ablations.tex',
         'All four ablations in one table: change against the original (positive panel = better, positive LBR = more '
         'exploitable, positive head-to-head = the changed arm wins)', 'experiments/ablation_*.json')


def table_leduc_mistakes():
    rank = {r[0]: r for r in read_csv(os.path.join(FINAL, 'leduc_h2h_ranking.csv'))[1:]}
    name = {'Deep CFR': 'deepcfr', 'DQN': 'dqn', 'NFSP': 'nfsp', 'PPO': 'ppo'}
    rows = []
    for lab, r in rank.items():
        with open(os.path.join(FINAL, 'hands', f'leduc_{name[lab]}_vs_cfr.analysis.json'), encoding='utf-8') as f:
            s = json.load(f)['summary']
        rows.append([lab, r[1], r[2], f'{s["mean_ev_loss"]:.3f}', f'{100 * s["share_with_loss_over_0.05bb"]:.0f}%',
                     f'{s["decisions"]}'])
    rows.sort(key=lambda r: -float(r[1]))
    _write_table(os.path.join(TABLES, 'leduc_mistakes'),
                 'Leduc: mistakes against the CFR+ equilibrium (400 deals, exact)',
                 ['algorithm', 'head-to-head vs others (mbb/hand)', 'exploitability (mbb/hand)',
                  'mean EV loss per decision (BB)', 'decisions losing over 0.05 BB', 'decisions'], rows)
    note('Tables', 'tables/leduc_mistakes.tex',
         'Leduc: mean EV loss per decision against CFR+ (exact, seed 1042 of each winner), next to exploitability and '
         'head-to-head', 'experiments/final/hands/leduc_*_vs_cfr.analysis.json, leduc_h2h_ranking.csv')


# --------------------------------------------------------------------------- #
# New figures
# --------------------------------------------------------------------------- #

def _save(fig, n, caption, source):
    fig.savefig(os.path.join(FIGURES, n + '.pdf'), bbox_inches='tight', facecolor=SURFACE)
    fig.savefig(os.path.join(FIGURES, n + '.png'), dpi=200, bbox_inches='tight', facecolor=SURFACE)
    note('Figures', f'figures/{n}.pdf', caption, source)


def _diverging():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('rgb', [DIV_NEG, DIV_MID, DIV_POS])


def _heatmap(ax, values, errs, rows, cols, vmax):
    v = np.array([[np.nan if x is None else x for x in r] for r in values], dtype=float)
    im = ax.imshow(np.ma.masked_invalid(v), cmap=_diverging(), vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_facecolor(SURFACE)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isnan(v[i, j]):
                ax.text(j, i, '–', ha='center', va='center', fontsize=8, color=MUTED)
                continue
            e = errs[i][j]
            sig = e is not None and abs(v[i, j]) > 2 * e
            ax.text(j, i, f'{v[i, j]:+.0f}' + (f'\n±{e:.0f}' if e is not None else ''), ha='center', va='center',
                    fontsize=7.5, color=INK, fontweight='bold' if sig else 'normal', linespacing=1.1)
    ax.set_xticks(range(len(cols)), cols, fontsize=8, color=INK2, rotation=30, ha='right')
    ax.set_yticks(range(len(rows)), rows, fontsize=8, color=INK2)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=2)
    ax.tick_params(which='both', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def figure_h2h_matrix(plt):
    cells = limit_cells()
    order = [r[0] for r in ranking()]
    vals = [[cells.get((a, b), (None, None))[0] for b in order] for a in order]
    errs = [[cells.get((a, b), (None, None))[1] for b in order] for a in order]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = _heatmap(ax, vals, errs, [label(k) for k in order], [label(k) for k in order], 300)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label('row wins (mbb/hand)', fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Limit hold'em head-to-head (row vs column, mean ± se over seed pairs; bold = beyond 2 se)",
                 fontsize=9, color=INK, loc='left')
    fig.tight_layout()
    _save(fig, 'h2h_matrix', "Limit hold'em round robin by algorithm, ordered by rank (blue = row wins, red = row loses; "
          'bold = more than 2 standard errors from zero)', 'experiments/final/h2h_algorithms.csv')
    plt.close(fig)


def figure_leduc_matrix(plt):
    rows = read_csv(os.path.join(FINAL, 'leduc_h2h_algorithms.csv'))
    cols = rows[0][1:]
    order = sorted(rows[1:], key=lambda r: -statistics.fmean(pm(v)[0] for v in r[1:-1] if v != '--'))
    vals = [[None if v == '--' else pm(v)[0] for v in r[1:]] for r in order]
    sds = [[None if v == '--' else float(re.search(r'sd (\d+)', v).group(1)) for v in r[1:]] for r in order]
    # exact values (no card noise): the only uncertainty is the seed spread, shown as "sd" below the value
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    im = _heatmap(ax, vals, [[None] * len(cols) for _ in order], [r[0] for r in order],
                  [c.replace(' equilibrium', '') for c in cols], 200)
    for i, r in enumerate(sds):
        for j, s in enumerate(r):
            if s is not None:
                ax.text(j, i + 0.28, f'sd {s:.0f}', ha='center', va='center', fontsize=6, color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label('row wins (mbb/hand)', fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    ax.set_title('Leduc head-to-head, exact (row vs column, mean over seed pairs; sd = seed spread)',
                 fontsize=9, color=INK, loc='left')
    fig.tight_layout()
    _save(fig, 'leduc_h2h_matrix', 'Leduc round robin by algorithm and against the CFR+ equilibrium, computed exactly '
          'over all deals (blue = row wins)', 'experiments/final/leduc_h2h_algorithms.csv')
    plt.close(fig)


def _marker_style(key):
    ''' Extended runs filled, standard runs hollow; colour and marker shape follow the algorithm. '''
    c = COLOR[base(key)]
    filled = key.endswith('_ext')
    return dict(marker=MARKER[base(key)], color=c, markerfacecolor=c if filled else SURFACE, markeredgecolor=c,
                linestyle='none')


def figure_ranking(plt):
    cells = limit_cells()
    rank = ranking()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    _axes_style(ax)
    ax.grid(True, axis='x', color=GRID, linewidth=0.6)
    ax.grid(False, axis='y')
    ax.axvline(0, color='#c3c2b7', linewidth=1)
    for i, (key, mean, _, sig) in enumerate(rank):
        y = len(rank) - 1 - i
        opp = [(b, cells[key, b]) for b in ORDER if (key, b) in cells]
        xs = [v for _, (v, _) in opp]
        ax.plot([min(xs), max(xs)], [y, y], color=GRID, linewidth=1.2, zorder=1)
        for b, (v, e) in opp:
            ax.plot(v, y, marker=MARKER[base(b)], markersize=4.5, color=MUTED, markerfacecolor=SURFACE,
                    markeredgewidth=0.9, linestyle='none', zorder=2)
        ax.plot(mean, y, markersize=10, markeredgewidth=1.6, zorder=3, **_marker_style(key))
        ax.annotate(f'{mean:+.0f}  ({sig} significant)', (mean, y), xytext=(0, 9), textcoords='offset points',
                    fontsize=7, color=INK2, ha='center')
    ax.set_yticks(range(len(rank)), [label(r[0]) for r in reversed(rank)], fontsize=8, color=INK2)
    ax.set_ylim(-0.6, len(rank) - 0.2)
    ax.set_xlabel('head-to-head win rate (mbb/hand); large marker = mean over the six opponents, '
                  'small = each opponent (shape = its algorithm)', fontsize=7.5)
    ax.set_title("Limit hold'em: ranking by the round robin", fontsize=10, color=INK, loc='left')
    fig.tight_layout()
    _save(fig, 'h2h_ranking', "Limit hold'em ranking: each algorithm's mean head-to-head result (large marker; filled = "
          'extended run) and its six matchups (small markers, shape = opponent algorithm; "significant" = beaten by more '
          'than 2 se, see h2h_matrix for the standard errors)',
          'experiments/final/h2h_algorithms.csv, h2h_ranking.csv')
    plt.close(fig)


def figure_measures(plt):
    res = agent_results()
    corr = read_csv(os.path.join(FINAL, 'measure_correlation.csv'))
    rho = {r[0]: r for r in corr[1:]}['head-to-head'][corr[0].index('-LBR')]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _axes_style(ax)
    for key in ORDER:
        pts = [r for r in res.values() if r['algo'] == key]
        if not pts:
            continue
        x, y = [p['lbr'] for p in pts], [p['h2h'] for p in pts]
        ax.plot(x, y, markersize=7, markeredgewidth=1.4, label=label(key), **_marker_style(key))
        top = max(range(len(y)), key=y.__getitem__)
        ax.annotate(label(key), (x[top], y[top]), xytext=(0, 8), textcoords='offset points',
                    fontsize=7, color=INK2, ha='center')
    ax.axhline(0, color='#c3c2b7', linewidth=1)
    ax.set_xlabel('LBR (mbb/hand, lower = less exploitable)', fontsize=8)
    ax.set_ylabel('head-to-head vs other algorithms (mbb/hand)', fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc='upper right', ncol=2)
    ax.set_title(f'Two measures, one order: head-to-head against LBR per agent (Spearman {rho})',
                 fontsize=9, color=INK, loc='left')
    fig.tight_layout()
    _save(fig, 'measures', 'Every agent (3 seeds per algorithm): mean head-to-head against the agents of the other '
          'algorithms against its LBR exploitability bound; filled = extended run',
          'experiments/final/results.json, measure_correlation.csv')
    plt.close(fig)


def figure_ablations(plt):
    items = []
    for name, algo, change, arms in ABLATIONS:
        m, h2h, seeds = ablation(name)
        for arm, how in arms:
            diff, (hm, hse) = arm_vs_a(m, h2h, seeds, arm)
            items.append((f'{LABEL[algo]}: {change}\n{arm} ({how})', algo, hm, hse, diff['lbr']))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True, gridspec_kw={'width_ratios': [1, 1]})
    for ax in (a1, a2):
        _axes_style(ax)
        ax.grid(False, axis='y')
        ax.axvline(0, color='#c3c2b7', linewidth=1)
    ys = list(range(len(items)))[::-1]
    for y, (_, algo, hm, hse, lbr) in zip(ys, items):
        a1.errorbar(hm, y, xerr=2 * hse, color=COLOR[algo], linewidth=2, capsize=0, zorder=2)
        a1.plot(hm, y, marker=MARKER[algo], markersize=8, color=COLOR[algo], markeredgecolor=SURFACE,
                markeredgewidth=1, linestyle='none', zorder=3)
        a1.annotate(f'{hm:+.0f} ± {hse:.0f}', (hm, y), xytext=(0, 8), textcoords='offset points', fontsize=7,
                    color=INK2, ha='center')
        a2.plot(lbr, [y] * len(lbr), marker=MARKER[algo], markersize=5, color=COLOR[algo],
                markerfacecolor=SURFACE, markeredgewidth=1, linestyle='none', zorder=2)
        a2.plot(statistics.fmean(lbr), y, marker='|', markersize=14, markeredgewidth=2, color=COLOR[algo], zorder=3)
        a2.annotate(f'{statistics.fmean(lbr):+.0f}', (statistics.fmean(lbr), y), xytext=(0, 8),
                    textcoords='offset points', fontsize=7, color=INK2, ha='center')
    a1.set_yticks(ys, [it[0] for it in items], fontsize=7.5, color=INK2)
    a1.set_ylim(-0.6, len(items) - 0.3)
    a1.set_xlabel('changed arm vs original, head-to-head\n(mbb/hand, mean ± 2 se; > 0 = change helps)', fontsize=7.5)
    a2.set_xlabel('LBR change per seed (mbb/hand; < 0 = less exploitable)\nhollow markers = seeds, bar = mean', fontsize=7.5)
    fig.suptitle('Ablations: each change against the original configuration (A), 3 seeds per arm', fontsize=10,
                 color=INK, x=0.01, ha='left')
    fig.tight_layout()
    _save(fig, 'ablations', 'The four ablations: head-to-head of the changed arm against the original (± 2 se) and the '
          'change in LBR per seed', 'experiments/ablation_*.json')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Further figures (a menu to choose from; not all are meant for the thesis)
# --------------------------------------------------------------------------- #

# two-category pairs (validated with dataviz validate_palette.js on the light surface; distinct from the algorithm colours)
PAIR_A, PAIR_B = '#6250d6', '#eb6834'   # violet, orange
BOTS = ['random', 'station', 'maniac', 'nit', 'tag', 'lag']
ALGO4 = ['dqn', 'ppo', 'nfsp', 'deepcfr']


def _label_ends(ax, x, y, text, dx=4, ha='left'):
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords='offset points', fontsize=7, color=INK2, va='center', ha=ha)


def _sd(text):
    m = re.search(r'sd (\d+(?:\.\d+)?)', text)
    return float(m.group(1)) if m else 0.0


def figure_rank_slope(plt):
    ''' Rank of the four algorithms in Leduc, in limit (standard budget) and in limit (longest run). '''
    cells = limit_cells()

    def limit_means(keys):
        return {base(k): statistics.fmean(cells[k, o][0] for o in keys if o != k) for k in keys}
    key_of = {v: k for k, v in LABEL.items()}
    leduc = {key_of[r[0]]: float(r[1]) for r in read_csv(os.path.join(FINAL, 'leduc_h2h_ranking.csv'))[1:]}
    columns = [('Leduc', leduc),
               ("Limit hold'em\n(standard budget)", limit_means(ALGO4)),
               ("Limit hold'em\n(extended where run)", limit_means(['dqn', 'ppo_ext', 'nfsp_ext', 'deepcfr_ext']))]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    _axes_style(ax)
    ax.grid(False)
    ranks = {}
    for j, (_, vals) in enumerate(columns):
        order = sorted(vals, key=lambda k: -vals[k])
        for r, k in enumerate(order):
            ranks[k, j] = r + 1
    for k in ALGO4:
        xs = list(range(len(columns)))
        ys = [ranks[k, j] for j in xs]
        ax.plot(xs, ys, color=COLOR[k], linewidth=2, marker=MARKER[k], markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1, zorder=3)
        for j, (_, vals) in enumerate(columns):
            ax.annotate(f'{vals[k]:+.0f}', (j, ys[j]), xytext=(0, -13), textcoords='offset points', fontsize=6.5,
                        color=INK2, ha='center')
        _label_ends(ax, xs[0], ys[0], LABEL[k], dx=-10, ha='right')
        _label_ends(ax, xs[-1], ys[-1], LABEL[k], dx=10)
    ax.set_xticks(range(len(columns)), [c[0] for c in columns], fontsize=8, color=INK2)
    ax.set_yticks([1, 2, 3, 4], ['1st', '2nd', '3rd', '4th'], fontsize=8, color=INK2)
    ax.set_ylim(4.5, 0.5)
    ax.set_xlim(-0.6, len(columns) - 0.4)
    for s in ('left', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title('Rank by head-to-head among the four algorithms (number = mean mbb/hand vs the other three)',
                 fontsize=9, color=INK, loc='left')
    fig.tight_layout()
    _save(fig, 'rank_slope', 'The ranking depends on the game and on the budget: rank of each algorithm by its mean '
          'head-to-head result against the other three, in Leduc and in limit hold\'em',
          'experiments/final/leduc_h2h_ranking.csv, h2h_algorithms.csv')
    plt.close(fig)


def figure_opponent_profile(plt):
    rows = {r[0]: r for r in read_csv(os.path.join(FINAL, 'algorithms.csv'))[1:]}
    header = read_csv(os.path.join(FINAL, 'algorithms.csv'))[0]
    order = [r[0] for r in ranking()]
    ys = np.arange(len(order))[::-1]
    # each agent's weakest matchup (lowest mean over seeds; the panel min is the mean of each seed's own minimum,
    # so it can differ slightly when the seeds' weakest bots differ)
    weakest = {k: min(BOTS, key=lambda b: pm(rows[k][header.index(f'vs {b}')])[0]) for k in order}
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
    for ax, bot in zip(axes.flat, BOTS):
        _axes_style(ax)
        ax.grid(False, axis='y')
        ax.axvline(0, color='#c3c2b7', linewidth=1)
        i = header.index(f'vs {bot}')
        vals = {k: pm(rows[k][i])[0] for k in order}
        worst = {k for k in order if weakest[k] == bot}
        for y, key in zip(ys, order):
            ax.errorbar(vals[key], y, xerr=_sd(rows[key][i]), color=COLOR[base(key)], linewidth=1.4, capsize=0,
                        zorder=2)
            ax.plot(vals[key], y, markersize=7, markeredgewidth=1.4, zorder=3, **_marker_style(key))
            ax.annotate(f'{vals[key]:+.0f}' + (' (min)' if key in worst else ''), (vals[key], y), xytext=(0, 7),
                        textcoords='offset points', fontsize=6, color=INK if key in worst else INK2, ha='center',
                        fontweight='bold' if key in worst else 'normal')
        ax.set_title(f'vs {bot}', fontsize=10, color=INK, loc='left')
        ax.set_ylim(-0.6, len(order) - 0.2)
    for ax in axes[:, 0]:
        ax.set_yticks(ys, [label(k) for k in order], fontsize=8, color=INK2)
    for ax in axes[1]:
        ax.set_xlabel('win rate (mbb/hand, ± sd over seeds)', fontsize=8)
    fig.suptitle('Held-out panel: win rate against each rule-based bot (filled = extended run; '
                 'bold "(min)" = the agent\'s weakest matchup)', fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    _save(fig, 'opponent_profile', 'Win rate of each algorithm against each of the six panel bots (2000 duplicate '
          'deals per bot, mean ± sd over seeds); "(min)" marks each agent\'s weakest matchup, which is what the panel '
          'min measures',
          'experiments/final/algorithms.csv')
    plt.close(fig)


def figure_cost(plt):
    cost = {COST_NAME[r[1]]: r for r in read_csv(os.path.join(FINAL, 'cost.csv'))[1:] if r[0] == 'limit'}
    rank = {r[0]: r[1] for r in ranking()}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    _axes_style(ax)
    ax.axhline(0, color='#c3c2b7', linewidth=1)
    for algo in ALGO4:
        pts = [(k, float(cost[k][5].split()[0]), _sd(cost[k][5]), rank[k]) for k in (algo, algo + '_ext') if k in cost]
        if len(pts) == 2:
            (_, h0, _, y0), (_, h1, _, y1) = pts
            ax.annotate('', (h1, y1), (h0, y0), arrowprops=dict(arrowstyle='->', color=COLOR[algo], linewidth=1.2,
                                                                 shrinkA=7, shrinkB=7))
        for k, h, sd, y in pts:
            ax.errorbar(h, y, xerr=sd, color=COLOR[algo], linewidth=1, capsize=0, zorder=2)
            ax.plot(h, y, markersize=9, markeredgewidth=1.6, zorder=3, **_marker_style(k))
            off = {'deepcfr': (-8, 12), 'nfsp_ext': (8, -12)}.get(k, (8, 0))
            ax.annotate(f'{label(k)}\n{cost[k][3]}', (h, y), xytext=off, textcoords='offset points', fontsize=6.5,
                        color=INK2, va='center', ha='right' if off[0] < 0 else 'left')
    ax.set_xscale('log')
    from final_report import _plain_log_axis
    _plain_log_axis(ax.xaxis)
    ax.set_xlim(0.7, 45)
    ax.set_xlabel('training wall time per run (hours, log scale, ± sd; 3-4 CPU threads)', fontsize=8)
    ax.set_ylabel('head-to-head vs the other six (mbb/hand)', fontsize=8)
    ax.set_title("Strength against training cost (arrow = standard to extended run)", fontsize=10, color=INK, loc='left')
    fig.tight_layout()
    _save(fig, 'cost_vs_strength', 'Mean head-to-head result against training wall time per run; arrows join the '
          'standard and the extended run of the same configuration (filled = extended)',
          'experiments/final/cost.csv, h2h_ranking.csv')
    plt.close(fig)


def figure_style_map(plt):
    rows = read_csv(os.path.join(FINAL, 'h2h_style.csv'))
    h = rows[0]
    by = {}
    for r in rows[1:]:
        by.setdefault(r[0], []).append({k: float(v.rstrip('%')) for k, v in zip(h[2:], r[2:])})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0))
    panels = [('VPIP', 'PFR', 'VPIP: hands played voluntarily (%)', 'PFR: hands raised pre-flop (%)', 'Pre-flop'),
              ('fold facing bet', 'AF post-flop', 'fold when facing a bet (%)', 'aggression factor post-flop',
               'Post-flop')]
    for ax, (kx, ky, lx, ly, title) in zip(axes, panels):
        _axes_style(ax)
        for key in ORDER:
            if key not in by:
                continue
            xs, ys = [d[kx] for d in by[key]], [d[ky] for d in by[key]]
            ax.plot(xs, ys, marker=MARKER[base(key)], markersize=3, color=COLOR[base(key)], alpha=0.35,
                    linestyle='none', zorder=1)
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            ax.plot(mx, my, markersize=9, markeredgewidth=1.6, zorder=3, **_marker_style(key))
            ax.annotate(label(key), (mx, my), xytext=(7, -11) if key.endswith('_ext') else (7, 4),
                        textcoords='offset points', fontsize=7, color=INK2)
        ax.set_xlabel(lx, fontsize=8)
        ax.set_ylabel(ly, fontsize=8)
        ax.set_title(title, fontsize=10, color=INK, loc='left')
    fig.suptitle('Playing style in the round robin (large = mean over opponents, small = each matchup; filled = extended)',
                 fontsize=10, color=INK, x=0.02, ha='left')
    fig.tight_layout()
    _save(fig, 'style_map', 'Style of each algorithm in the head-to-head matches: looseness against pre-flop '
          'aggression, and folding against post-flop aggression', 'experiments/final/h2h_style.csv')
    plt.close(fig)


def _action_curve(dirs):
    ''' steps, {action: mean frequency over seeds} at the evaluation points all seeds share. '''
    data = [np.atleast_1d(np.genfromtxt(os.path.join(d, 'panel.csv'), delimiter=',', names=True)) for d in dirs]
    common = sorted(set.intersection(*[set((x['episode'] + 1).tolist()) for x in data]))
    out = {}
    for act in ('fold', 'check', 'call', 'raise'):
        vals = [[dict(zip((x['episode'] + 1).tolist(), x[f'act_{act}'].tolist()))[c] for c in common] for x in data]
        out[act] = np.mean(vals, axis=0)
    return np.array(common), out


def _action_series():
    ''' [(algo, dashed, x as a multiple of the standard budget, {action: % of decisions})] '''
    from final_report import ALGOS as FR_ALGOS, EXTENDED, runs, config
    out = []
    for algo in ALGO4:
        study = FR_ALGOS[algo][0]
        budget = config(study)['budget']
        for dirs, dashed in ((runs(study), False), (runs(study, EXTENDED[algo]) if algo in EXTENDED else [], True)):
            if not dirs:
                continue
            x, c = _action_curve(dirs)
            if dashed:   # identical to the standard run up to its budget; beyond it, smoothed (evaluated more often)
                keep = x >= budget
                k = 5
                c = {act: np.convolve(np.pad(v, (k // 2, k // 2), mode='edge'), np.ones(k) / k, 'valid')[keep]
                     for act, v in c.items()}
                x = x[keep]
            out.append((algo, dashed, x / budget, {a: 100 * v for a, v in c.items()}))
    return out


def _draw_action(ax, series, act, markersize=3.5):
    from final_report import _plain_log_axis
    _axes_style(ax)
    for algo, dashed, x, c in series:
        ax.plot(x, c[act], '--' if dashed else '-', color=COLOR[algo], linewidth=1.6 if markersize < 4 else 2,
                marker=MARKER[algo], markersize=markersize, markevery=max(1, len(x) // 6),
                markeredgecolor=SURFACE, markeredgewidth=0.6, label=None if dashed else LABEL[algo])
    ax.set_xscale('log')
    _plain_log_axis(ax.xaxis)
    ax.axvline(1, color='#c3c2b7', linewidth=1)


def _end_labels(ax, series, act):
    ''' One label per algorithm in a column right of the longest run, spread to a minimum vertical gap,
    each joined to the end of its line by a thin leader. '''
    last = {}
    for algo, _, x, c in series:   # the extended run comes after the standard one, so it wins
        last[algo] = (x[-1], c[act][-1])
    lo, hi = ax.get_ylim()
    gap = 0.06 * (hi - lo)
    placed = []
    for algo, (x, y) in sorted(last.items(), key=lambda t: t[1][1]):
        placed.append([algo, x, y, max(y, placed[-1][3] + gap) if placed else y])
    shift = max(0.0, placed[-1][3] - (hi - gap / 2))
    x_text = max(p[1] for p in placed) * 1.15
    for algo, x, y, yt in placed:
        ax.annotate(LABEL[algo], (x, y), xytext=(x_text, yt - shift), textcoords='data', fontsize=7.5,
                    color=INK2, va='center', ha='left',
                    arrowprops=dict(arrowstyle='-', color='#c3c2b7', linewidth=0.8, shrinkA=2, shrinkB=3))
    ax.set_ylim(lo, hi)


def figure_style_training(plt):
    acts = ['fold', 'check', 'call', 'raise']
    series = _action_series()
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=False)
    for ax, act in zip(axes, acts):
        _draw_action(ax, series, act)
        ax.set_title(f'{act} (% of decisions)', fontsize=9, color=INK, loc='left')
        ax.set_xlabel('training progress (× standard budget, log)', fontsize=7.5)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('Action frequencies against the panel during training (solid = standard run, dashed = extended run '
                 'past the standard budget, rolling mean of 5)', fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    _save(fig, 'actions_over_training', 'How often each algorithm folds, checks, calls and raises against the panel '
          'bots over training (mean over 3 seeds; x = training steps divided by the standard budget)',
          'experiments/tune/<study>/finalists|extended_*/trial_*/seed_*/panel.csv')
    plt.close(fig)

    # one chart per action
    for act in acts:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        _draw_action(ax, series, act, markersize=5)
        ax.set_xlim(right=ax.get_xlim()[1] * 1.9)   # room for the end labels
        _end_labels(ax, series, act)
        ax.set_ylabel(f'{act} (% of decisions against the panel)', fontsize=8)
        ax.set_xlabel('training progress (multiple of the standard budget, log scale; grey line = standard budget)',
                      fontsize=8)
        ax.set_title(f'How often each algorithm chooses to {act} during training (solid = standard run, '
                     f'dashed = extended)', fontsize=9, color=INK, loc='left')
        fig.tight_layout()
        _save(fig, f'actions_over_training_{act}', f'Share of {act} decisions against the panel bots over training '
              '(mean over 3 seeds; extended runs past the standard budget smoothed with a rolling mean of 5)',
              'experiments/tune/<study>/finalists|extended_*/trial_*/seed_*/panel.csv')
        plt.close(fig)


def figure_endings(plt):
    rows = read_csv(os.path.join(FINAL, 'h2h_endings.csv'))[1:]
    sd, nsd = {}, {}
    for r in rows:
        a, b = r[0].split(' vs ')
        s, n = float(r[4]), float(r[5])
        sd.setdefault(a, []).append(s); nsd.setdefault(a, []).append(n)
        sd.setdefault(b, []).append(-s); nsd.setdefault(b, []).append(-n)
    order = [r[0] for r in ranking()]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    _axes_style(ax)
    ax.grid(False, axis='y')
    ax.axvline(0, color='#c3c2b7', linewidth=1)
    ys = np.arange(len(order))[::-1]
    h = 0.36
    s_vals = [statistics.fmean(sd[k]) for k in order]
    n_vals = [statistics.fmean(nsd[k]) for k in order]
    ax.barh(ys + h / 2 + 0.01, s_vals, height=h, color=PAIR_A, label='won at showdown', edgecolor=SURFACE, linewidth=1)
    ax.barh(ys - h / 2 - 0.01, n_vals, height=h, color=PAIR_B, label='won when someone folded', edgecolor=SURFACE,
            linewidth=1)
    for y, s, n in zip(ys, s_vals, n_vals):
        ax.plot(s + n, y, marker='|', markersize=16, markeredgewidth=2, color=INK, zorder=3)
        ax.annotate(f'{s + n:+.0f}', (s + n, y), xytext=(0, 11), textcoords='offset points', fontsize=6.5,
                    color=INK, ha='center')
    ax.plot([], [], marker='|', markersize=10, markeredgewidth=2, color=INK, linestyle='none', label='total')
    ax.set_yticks(ys, [label(k) for k in order], fontsize=8, color=INK2)
    ax.set_xlabel('mean over the six opponents (mbb/hand)', fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax.set_title('Where the head-to-head result comes from: showdowns against folds', fontsize=10, color=INK, loc='left')
    fig.tight_layout()
    _save(fig, 'h2h_endings', "Each algorithm's head-to-head result split into what it wins at showdown and what it "
          'wins when a player folds (the two add up to the total)', 'experiments/final/h2h_endings.csv')
    plt.close(fig)


def _leduc_decisions():
    out = {}
    for k in ALGO4:
        p = os.path.join(OUT, 'data', f'leduc_{k}_decisions.json')
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as f:
                out[k] = json.load(f)['decisions']
    return out


def figure_leduc_losses(plt):
    dec = _leduc_decisions()
    if not dec:
        print('  leduc decisions missing (hand_analysis.py leduc --decisions); skipped leduc_losses')
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={'width_ratios': [1.3, 1]})
    _axes_style(a1)
    grid = np.logspace(-3, np.log10(3), 200)
    for k in ALGO4:
        loss = np.array([d['ev_loss'] for d in dec[k]])
        share = [(loss > g).mean() * 100 for g in grid]
        a1.plot(grid, share, color=COLOR[k], linewidth=2, marker=MARKER[k], markersize=4, markevery=25,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=LABEL[k])
    a1.set_xscale('log')
    import matplotlib.ticker as ticker
    a1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    a1.xaxis.set_minor_formatter(ticker.NullFormatter())
    a1.legend(fontsize=7, frameon=False, loc='upper right')
    a1.set_xlabel('EV loss of the decision (big blinds, log scale)', fontsize=8)
    a1.set_ylabel('decisions losing more than this (%)', fontsize=8)
    a1.set_title('Share of decisions above a given loss', fontsize=9, color=INK, loc='left')
    _axes_style(a2)
    a2.grid(False, axis='x')
    width = 0.2
    for i, k in enumerate(ALGO4):
        for j, street in enumerate(('round 1', 'round 2')):
            l = [d['ev_loss'] for d in dec[k] if d['street'] == street]
            total = sum(l) / 400   # per deal over all 400 deals (incl. those the opponent folded at once)
            a2.bar(j + (i - 1.5) * width, total, width=width - 0.02, color=COLOR[k], edgecolor=SURFACE,
                   linewidth=1, label=LABEL[k] if j == 0 else None)
    a2.set_xticks([0, 1], ['round 1 (pre-board)', 'round 2 (board card shown)'], fontsize=8, color=INK2)
    a2.set_ylabel('EV loss per deal (big blinds)', fontsize=8)
    a2.legend(fontsize=7, frameon=False)
    a2.set_title('Where the loss is made', fontsize=9, color=INK, loc='left')
    fig.suptitle('Leduc: mistakes against the CFR+ equilibrium over 400 deals (exact)', fontsize=10, color=INK,
                 x=0.01, ha='left')
    fig.tight_layout()
    _save(fig, 'leduc_losses', 'Leduc: distribution of the per-decision EV loss against CFR+ (exact, 400 deals, seed '
          '1042 of each winner) and the loss per deal by betting round',
          'experiments/thesis/data/leduc_*_decisions.json (hand_analysis.py leduc --decisions)')
    plt.close(fig)


def figure_seeds(plt):
    rows = read_csv(os.path.join(FINAL, 'agents.csv'))
    h = rows[0]
    by = {}
    for r in rows[1:]:
        by.setdefault(r[0].rsplit('_s', 1)[0], []).append((r[0].rsplit('_s', 1)[1], float(r[h.index('panel min')]),
                                                           pm(r[h.index('LBR')])[0]))
    order = [r[0] for r in ranking()]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (col, xl) in zip(axes, ((1, 'held-out panel min (mbb/hand, higher = better)'),
                                    (2, 'LBR (mbb/hand, lower = less exploitable)'))):
        _axes_style(ax)
        ax.grid(False, axis='y')
        for i, key in enumerate(order):
            y = len(order) - 1 - i
            vals = [p[col] for p in by[key]]
            ax.plot([min(vals), max(vals)], [y, y], color=COLOR[base(key)], linewidth=1.2, alpha=0.5, zorder=1)
            ax.plot(vals, [y] * len(vals), markersize=6, markeredgewidth=1.2, zorder=2, **_marker_style(key))
            ax.plot(statistics.fmean(vals), y, marker='|', markersize=14, markeredgewidth=2, color=INK, zorder=3)
        ax.set_xlabel(xl, fontsize=8)
    axes[0].set_yticks(range(len(order)), [label(k) for k in reversed(order)], fontsize=8, color=INK2)
    fig.suptitle('Seed to seed: each marker is one training seed (1042-1044), bar = mean', fontsize=10, color=INK,
                 x=0.01, ha='left')
    fig.tight_layout()
    _save(fig, 'seed_spread', 'Spread over the three training seeds of every algorithm: held-out panel min and LBR',
          'experiments/final/agents.csv')
    plt.close(fig)


def figure_importance(plt):
    studies = [('dqn', 'dqn_v1', 'leduc_dqn_v1'), ('ppo', 'ppo_v2', 'leduc_ppo_v1'),
               ('nfsp', 'nfsp_v3', 'leduc_nfsp_v1'), ('deepcfr', 'deepcfr_v1', 'leduc_deepcfr_v1')]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6))
    for j, (algo, lim, led) in enumerate(studies):
        for i, (study, game) in enumerate(((lim, "limit"), (led, 'Leduc'))):
            ax = axes[i, j]
            _axes_style(ax)
            ax.grid(False, axis='y')
            rows = read_csv(os.path.join(FINAL, 'tuning', f'{study}.csv'))
            col = next(c for c in range(len(rows[0])) if 'trials at evaluation' in rows[0][c])
            vals = sorted([(r[0], float(r[col])) for r in rows[1:] if r[col]], key=lambda t: t[1])
            ys = np.arange(len(vals))
            ax.barh(ys, [v for _, v in vals], color=COLOR[algo], height=0.65, edgecolor=SURFACE, linewidth=1)
            for y, (_, v) in zip(ys, vals):
                ax.annotate(f'{v:.2f}', (v, y), xytext=(3, 0), textcoords='offset points', fontsize=6.5,
                            color=INK2, va='center')
            ax.set_yticks(ys, [n for n, _ in vals], fontsize=7, color=INK2)
            ax.set_xlim(0, 1.05)
            ax.set_title(f'{LABEL[algo]}, {game}', fontsize=9, color=INK, loc='left')
            if i == 1:
                ax.set_xlabel('importance (fANOVA, sums to 1)', fontsize=7.5)
    fig.suptitle('Which hyperparameters mattered: fANOVA importance over all tuning trials at an early evaluation',
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    _save(fig, 'tuning_importance', 'Hyperparameter importance (fANOVA) of every tuning study, from all trials at '
          'the same early evaluation (the completed trials alone are too few for a stable estimate)',
          'experiments/final/tuning/*.csv')
    plt.close(fig)


def figure_win_rate(plt):
    ''' Head-to-head win rate per algorithm (bar = mean, dots = the three seeds), limit and Leduc. '''
    res = agent_results()
    limit = {k: [r['h2h'] for r in res.values() if r['algo'] == k] for k in ORDER}
    rows = read_csv(os.path.join(FINAL, 'leduc_h2h_agents.csv'))
    cols = rows[0][1:]
    leduc = {}
    for r in rows[1:]:
        algo = r[0].rsplit('_s', 1)[0]
        vals = [float(v) for c, v in zip(cols, r[1:]) if v not in ('--', '') and c != 'CFR+'
                and c.rsplit('_s', 1)[0] != algo]
        leduc.setdefault(algo, []).append(statistics.fmean(vals))
    # the means must reproduce the published rankings
    for k, m, _, _ in ranking():
        assert abs(statistics.fmean(limit[k]) - m) < 1, (k, statistics.fmean(limit[k]), m)
    key_of = {v: k for k, v in LABEL.items()}
    for r in read_csv(os.path.join(FINAL, 'leduc_h2h_ranking.csv'))[1:]:
        assert abs(statistics.fmean(leduc[key_of[r[0]]]) - float(r[1])) < 1, r

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={'width_ratios': [7, 4.6]})
    for ax, data, title in ((a1, limit, "Limit hold'em"), (a2, leduc, 'Leduc (exact)')):
        _axes_style(ax)
        ax.grid(False, axis='x')
        order = sorted(data, key=lambda k: -statistics.fmean(data[k]))
        x = np.arange(len(order))
        for i, k in enumerate(order):
            m = statistics.fmean(data[k])
            c = COLOR[base(k)]
            ext = k.endswith('_ext') or ax is a2   # Leduc has only one run per algorithm: drawn filled
            ax.bar(i, m, width=0.62, color=c if ext else SURFACE, edgecolor=c, linewidth=0 if ext else 1.8, zorder=2)
            ax.plot([i] * len(data[k]), data[k], marker=MARKER[base(k)], markersize=4.5, linestyle='none',
                    color=INK, markerfacecolor=SURFACE, markeredgewidth=0.9, zorder=3)
            end = max([m] + data[k]) if m >= 0 else min([m] + data[k])   # beyond the bar and every seed
            ax.annotate(f'{m:+.0f}', (i, end), xytext=(0, 5 if m >= 0 else -5), textcoords='offset points',
                        fontsize=7.5, color=INK, ha='center', va='bottom' if m >= 0 else 'top')
        ax.axhline(0, color='#c3c2b7', linewidth=1, zorder=1)
        ax.set_xticks(x, [label(k) for k in order], fontsize=8, color=INK2, rotation=25, ha='right')
        ax.set_title(title, fontsize=10, color=INK, loc='left')
        ax.margins(y=0.12)
    a1.set_ylabel('head-to-head win rate vs the other algorithms\n(mbb/hand)', fontsize=8)
    fig.suptitle('Win rate by algorithm (bar = mean over 3 seeds, dots = each seed; hollow = standard run, '
                 'filled = extended)', fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    _save(fig, 'win_rate', "Each algorithm's mean head-to-head win rate against the agents of all other algorithms, "
          "in limit hold'em (duplicate deals) and Leduc (exact); dots = the three training seeds",
          'experiments/final/results.json, leduc_h2h_agents.csv')
    plt.close(fig)


def figure_position(plt):
    rows = read_csv(os.path.join(FINAL, 'h2h_position.csv'))
    cols = rows[0][1:]
    sb, bb = {}, {}
    for r in rows[1:]:
        for c, v in zip(cols, r[1:]):
            if v != '--':
                s, b = (float(x) for x in v.split(' / '))
                sb.setdefault(r[0], []).append(s)
                bb.setdefault(r[0], []).append(b)
    order = [r[0] for r in ranking()]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    _axes_style(ax)
    ax.grid(False, axis='y')
    ax.axvline(0, color='#c3c2b7', linewidth=1)
    ys = np.arange(len(order))[::-1]
    for y, k in zip(ys, order):
        s, b = statistics.fmean(sb[k]), statistics.fmean(bb[k])
        ax.plot([s, b], [y, y], color=GRID, linewidth=3, zorder=1)
        ax.plot(s, y, 'o', color=PAIR_A, markersize=8, markeredgecolor=SURFACE, zorder=2)
        ax.plot(b, y, 's', color=PAIR_B, markersize=8, markeredgecolor=SURFACE, zorder=2)
    ax.plot([], [], 'o', color=PAIR_A, label='as small blind (acts last after the flop)')
    ax.plot([], [], 's', color=PAIR_B, label='as big blind')
    ax.set_yticks(ys, [label(k) for k in order], fontsize=8, color=INK2)
    ax.set_xlabel('head-to-head vs the other six (mbb/hand)', fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    ax.set_title('Head-to-head result by position', fontsize=10, color=INK, loc='left')
    fig.tight_layout()
    _save(fig, 'h2h_position', 'Mean head-to-head result of each algorithm as small blind and as big blind (every deal '
          'is played from both seats; the overall result is the average)', 'experiments/final/h2h_position.csv')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Findings figures: measures across games, folding discipline, tuning, preflop, Leduc leaks
# --------------------------------------------------------------------------- #

FOLD_COLOR = '#d9d8d2'   # fold = not playing the hand: neutral, so raise / call carry the hue
DATA = os.path.join(OUT, 'data')


def _spearman(a, b):
    from final_evaluation import _ranks
    return float(np.corrcoef(_ranks(a), _ranks(b))[0, 1])


def _leduc_agents():
    ''' [(algo, seed, run dir, exploitability mbb/hand)] of the Leduc winners. '''
    from final_report import ALGOS as FR_ALGOS, runs
    out = []
    for algo in ALGO4:
        study = FR_ALGOS[algo][1]
        with open(os.path.join(EXP, 'tune', study, 'finalists', 'finalists.json'), encoding='utf-8') as f:
            record = json.load(f)
        w = record['winner']
        for d in runs(study):
            seed = int(d.rsplit('seed_', 1)[1])
            run = next(r for r in record['runs'] if r['trial'] == w['trial'] and r['seed'] == seed)
            out.append((algo, seed, d, -run['final_min']))
    return out


def figure_measures_games(plt):
    ''' Limit: head-to-head vs LBR; Leduc: head-to-head vs exact exploitability, per agent. '''
    res = agent_results()
    rows = read_csv(os.path.join(FINAL, 'leduc_h2h_agents.csv'))
    cols = rows[0][1:]
    leduc_h2h = {}
    for r in rows[1:]:
        algo = r[0].rsplit('_s', 1)[0]
        vals = [float(v) for c, v in zip(cols, r[1:]) if v not in ('--', '') and c != 'CFR+'
                and c.rsplit('_s', 1)[0] != algo]
        leduc_h2h[r[0]] = statistics.fmean(vals)
    leduc = [(algo, leduc_h2h[f'{algo}_s{seed}'], expl) for algo, seed, _, expl in _leduc_agents()]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    rho_l = _spearman([r['h2h'] for r in res.values()], [-r['lbr'] for r in res.values()])
    rho_d = _spearman([h for _, h, _ in leduc], [-e for _, _, e in leduc])
    for ax in (a1, a2):
        _axes_style(ax)
        ax.axhline(0, color='#c3c2b7', linewidth=1)
    for key in ORDER:
        pts = [r for r in res.values() if r['algo'] == key]
        x, y = [p['lbr'] for p in pts], [p['h2h'] for p in pts]
        a1.plot(x, y, markersize=7, markeredgewidth=1.4, **_marker_style(key))
        top = max(range(len(y)), key=y.__getitem__)
        a1.annotate(label(key), (x[top], y[top]), xytext=(0, 8), textcoords='offset points', fontsize=7,
                    color=INK2, ha='center')
    for algo in ALGO4:
        pts = [(e, h) for a, h, e in leduc if a == algo]
        x, y = [p[0] for p in pts], [p[1] for p in pts]
        a2.plot(x, y, marker=MARKER[algo], markersize=8, color=COLOR[algo], markeredgecolor=SURFACE,
                markeredgewidth=1, linestyle='none')
        top = max(range(len(y)), key=y.__getitem__)
        a2.annotate(LABEL[algo], (x[top], y[top]), xytext=(0, 9), textcoords='offset points', fontsize=7.5,
                    color=INK2, ha='center')
    a1.set_xlabel('LBR lower bound on exploitability (mbb/hand)', fontsize=8)
    a2.set_xlabel('exact exploitability (mbb/hand)', fontsize=8)
    a1.set_ylabel('head-to-head vs other algorithms (mbb/hand)', fontsize=8)
    a1.set_title(f"Limit hold'em: the measures agree (Spearman {rho_l:+.2f})", fontsize=9.5, color=INK, loc='left')
    a2.set_title(f'Leduc: they agree only weakly (Spearman {rho_d:+.2f})', fontsize=9.5, color=INK, loc='left')
    a1.margins(y=0.1)
    a2.margins(y=0.1)
    fig.suptitle('Beating the other agents against being hard to exploit, every agent (further left = less '
                 'exploitable, higher = wins more)', fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    _save(fig, 'measures_by_game', "Head-to-head win rate against exploitability for every agent: in limit hold'em "
          '(LBR) the two orders agree closely, in Leduc (exact) only weakly: DQN wins head-to-head although it is '
          'the most exploitable agent; Spearman over the agents', 'experiments/final/results.json, leduc_h2h_agents.csv, '
          'tune/leduc_*/finalists/finalists.json')
    plt.close(fig)


def _wilson(p, n, z=1.96):
    if n <= 0:
        return 0.0, 0.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def figure_fold_discipline(plt):
    with open(os.path.join(FINAL, 'results.json'), encoding='utf-8') as f:
        tasks = json.load(f)['tasks']
    metrics = [('fold when equity < pot odds', 'Folds when behind\n(equity below the pot odds; higher = better)'),
               ('fold when equity >> pot odds', 'Folds when well ahead\n(equity 0.15 above the pot odds; lower = better)')]
    agg = {}
    for k, v in tasks.items():
        if not k.startswith('style:'):
            continue
        _, a, b = k.split(':')
        for side, name in (('a', a), ('b', b)):
            for m, _ in metrics:
                x = v[side].get(m)
                if x and x['value'] is not None:
                    s = agg.setdefault((name.rsplit('_s', 1)[0], name, m), [0.0, 0])
                    s[0] += x['value'] * x['n']
                    s[1] += x['n']
    order = [r[0] for r in ranking()]
    seeds = {k[0]: k[1].rsplit('_s', 1)[1] for k in agg}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    ys = np.arange(len(order))[::-1]
    for ax, (m, title) in zip(axes, metrics):
        _axes_style(ax)
        ax.grid(False, axis='y')
        for y, key in zip(ys, order):
            s, n = agg[(key, f'{key}_s{seeds[key]}', m)]
            p = s / n
            lo, hi = _wilson(p, n)
            ax.plot([100 * lo, 100 * hi], [y, y], color=COLOR[base(key)], linewidth=2, zorder=2)
            ax.plot(100 * p, y, markersize=8, markeredgewidth=1.5, zorder=3, **_marker_style(key))
            ax.annotate(f'{100 * p:.0f}%  (n={n})', (100 * hi, y), xytext=(5, 0), textcoords='offset points',
                        fontsize=6.5, color=INK2, va='center')
        ax.set_title(title, fontsize=9, color=INK, loc='left')
        ax.set_xlabel('% of such decisions folded (95% interval)', fontsize=8)
        ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.25)
    axes[0].set_yticks(ys, [f'{label(k)} ({seeds[k]})' for k in order], fontsize=8, color=INK2)
    fig.suptitle('Folding discipline in the round robin (one seed per algorithm)', fontsize=10, color=INK,
                 x=0.01, ha='left')
    fig.tight_layout()
    _save(fig, 'fold_discipline', 'How often each agent folds when its equity is below the pot odds (a correct fold) '
          'and when it is well above them (a mistake), pooled over its round-robin matches; one seed per algorithm '
          '(the style statistics were recorded for one seed); 95% Wilson intervals on the number of decisions',
          'experiments/final/results.json (style:*)')
    plt.close(fig)


def figure_nfsp_divergence(plt):
    import final_report as fr
    from optuna.trial import TrialState
    data = {}
    for study in ('nfsp_v2', 'nfsp_v3'):
        pts = []
        for t in fr._load_study_readonly(study).get_trials(deepcopy=False):
            p = os.path.join(fr.TUNE, study, 'trials', f'trial_{t.number}', 'panel.csv')
            if not (t.params and t.intermediate_values and os.path.isfile(p)):
                continue
            d = np.atleast_1d(np.genfromtxt(p, delimiter=',', names=True))
            done = t.state == TrialState.COMPLETE
            score = t.value if done else t.intermediate_values[max(t.intermediate_values)]
            pts.append((t.params['rl_learning_rate'], score, 100 * float(d['act_fold'][-1]), done))
        data[study] = pts
    styles = {'nfsp_v2': (PAIR_A, 'o', 'nfsp_v2: SGD, no divergence check'),
              'nfsp_v3': (PAIR_B, 's', 'nfsp_v3: Adam, clipping, divergence check')}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    for ax in (a1, a2):
        _axes_style(ax)
    for study, pts in data.items():
        c, mk, lab = styles[study]
        for lr, s, fold, done in pts:
            kw = dict(marker=mk, markersize=6, color=c, markerfacecolor=c if done else SURFACE,
                      markeredgewidth=1.2, linestyle='none')
            a1.plot(lr, s, **kw)
            a2.plot(fold, s, **kw)
        a1.plot([], [], marker=mk, color=c, linestyle='none', label=lab)
    a1.plot([], [], marker='o', color=INK2, linestyle='none', label='complete trial (filled)')
    a1.plot([], [], marker='o', color=INK2, markerfacecolor=SURFACE, linestyle='none',
            label='pruned trial, last score (hollow)')
    a1.set_xscale('log')
    a1.set_xlabel('RL (best-response) learning rate, log scale', fontsize=8)
    a2.set_xlabel('fold rate against the panel at the trial\'s last evaluation (%)', fontsize=8)
    a1.set_ylabel('tuning score (panel min, mbb/hand)', fontsize=8)
    a1.legend(fontsize=6.5, frameon=False, loc='lower left')
    a1.set_title('Large learning rates scored best', fontsize=9.5, color=INK, loc='left')
    a2.set_title('...because the diverged agents stopped folding', fontsize=9.5, color=INK, loc='left')
    fig.suptitle('NFSP tuning rewarded a diverged Q-network (all 8 checked nfsp_v2 finalists had NaN weights)',
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    _save(fig, 'nfsp_divergence', 'Every NFSP tuning trial: score against the RL learning rate and against the fold '
          'rate. In nfsp_v2 the large learning rates diverged the best-response Q-network to NaN, the agent then '
          'never folded, and that scored better on the panel than the working nfsp_v3 agents; pruned trials show '
          'their last (earlier) score', 'experiments/tune/nfsp_v2|v3 (optuna journal, trials/*/panel.csv)')
    plt.close(fig)


def figure_selection(plt):
    ''' Training-time score (tuning deals) of every finalist run against its held-out panel min. '''
    from final_report import ALGOS as FR_ALGOS
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    _axes_style(ax)
    xs_all, ys_all = [], []
    for algo in ALGO4:
        study = FR_ALGOS[algo][0]
        with open(os.path.join(EXP, 'tune', study, 'finalists', 'finalists.json'), encoding='utf-8') as f:
            record = json.load(f)
        w = record['winner']['trial']
        for r in record['runs']:
            p = os.path.join(REPO, r['log_dir'], 'heldout.json')
            if 'training_score' not in r or not os.path.isfile(p):
                continue
            with open(p, encoding='utf-8') as f:
                y = json.load(f)['panel']['min']
            x = r['training_score']
            xs_all.append(x)
            ys_all.append(y)
            ax.plot(x, y, marker=MARKER[algo], markersize=7, color=COLOR[algo], markeredgewidth=1.3,
                    markerfacecolor=COLOR[algo] if r['trial'] == w else SURFACE, linestyle='none', zorder=3)
        ax.plot([], [], marker=MARKER[algo], color=COLOR[algo], linestyle='none', label=LABEL[algo])
    lo, hi = min(xs_all + ys_all) - 50, max(xs_all + ys_all) + 50
    ax.plot([lo, hi], [lo, hi], color='#c3c2b7', linewidth=1, linestyle=':', zorder=1)
    ax.annotate('same score', (hi, hi), xytext=(-4, -10), textcoords='offset points', fontsize=6.5, color=MUTED,
                ha='right')
    rho = _spearman(xs_all, ys_all)
    ax.set_xlabel('training-time score (mean of the last 3 panel evaluations, tuning deals)', fontsize=8)
    ax.set_ylabel('held-out panel min (new deals, mbb/hand)', fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax.set_title(f'Did the tuning signal carry over to new deals? (Spearman {rho:+.2f}, {len(xs_all)} runs)',
                 fontsize=9.5, color=INK, loc='left')
    fig.tight_layout()
    _save(fig, 'selection_heldout', 'Every finalist run (top 3 trials x 3 new seeds per algorithm): its training-time '
          'panel score against its panel min on held-out deals; filled = the chosen winner',
          'experiments/tune/<study>/finalists/finalists.json, */heldout.json')
    plt.close(fig)


# ---- preflop charts -------------------------------------------------------- #

RANKS13 = 'AKQJT98765432'


def _hand_class(c1, c2):
    ''' 'AKs' / 'AKo' / 'AA' and its (row, column) in the 13x13 chart (suited above the diagonal). '''
    r1, r2 = RANKS13.index(c1[1]), RANKS13.index(c2[1])
    hi, lo = min(r1, r2), max(r1, r2)
    if r1 == r2:
        return hi, hi
    return (hi, lo) if c1[0] == c2[0] else (lo, hi)


def _preflop_probs():
    ''' {agent key: {situation: 13x13x4 action probabilities}} averaged over combos and seeds (cached). '''
    path = os.path.join(DATA, 'preflop.json')
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            return {k: {s: np.array(v) for s, v in d.items()} for k, d in json.load(f).items()}
    import rlcard_patches  # noqa: F401
    import rlcard
    from agent_loading import load_policy
    from final_report import ALGOS as FR_ALGOS, EXTENDED, runs
    env = rlcard.make('limit-holdem', config={'seed': 0})
    s_open, _ = env.reset()
    s_bb, _ = env.step(1)                   # the small blind raises; the big blind decides
    cards = list(env.card2index)
    combos = [(a, b) for i, a in enumerate(cards) for b in cards[i + 1:]]
    cells = [_hand_class(a, b) for a, b in combos]
    out = {}
    for key in ORDER:
        algo = base(key)
        study = FR_ALGOS[algo][0]
        dirs = runs(study, EXTENDED[algo]) if key.endswith('_ext') else runs(study)
        acc = {'sb_open': np.zeros((13, 13, 4)), 'bb_vs_raise': np.zeros((13, 13, 4))}
        cnt = np.zeros((13, 13))
        for a, b in combos:
            cnt[_hand_class(a, b)] += 1
        for d in dirs:
            pol = load_policy(algo, d)
            for sit, st in (('sb_open', s_open), ('bb_vs_raise', s_bb)):
                base_obs = np.asarray(st[pol.obs_key], dtype=np.float32).copy()
                for c in st['raw_obs']['hand']:
                    base_obs[env.card2index[c]] = 0.0
                obs = np.repeat(base_obs[None], len(combos), axis=0)
                for i, (a, b) in enumerate(combos):
                    obs[i, env.card2index[a]] = obs[i, env.card2index[b]] = 1.0
                p = pol.probs_batch_obs(obs, list(st['legal_actions']))
                for (r, c), pr in zip(cells, p):
                    acc[sit][r, c] += pr
        out[key] = {s: (v / (cnt[..., None] * len(dirs))).tolist() for s, v in acc.items()}
        print(f'  preflop: {key} ({len(dirs)} seeds)')
    os.makedirs(DATA, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    return {k: {s: np.array(v) for s, v in d.items()} for k, d in out.items()}


def figure_preflop(plt):
    from matplotlib.patches import Rectangle
    probs = _preflop_probs()
    combos_per = np.array([[6 if i == j else (4 if i < j else 12) for j in range(13)] for i in range(13)])
    titles = {'sb_open': 'Small blind, first to act pre-flop', 'bb_vs_raise': 'Big blind facing a pre-flop raise'}
    for sit, sit_title in titles.items():
        fig, axes = plt.subplots(2, 4, figsize=(12.5, 7.4), gridspec_kw={'hspace': 0.45})
        for ax, key in zip(axes.flat, [r[0] for r in ranking()]):
            p = probs[key][sit]          # [..., call, raise, fold, check]
            for i in range(13):
                for j in range(13):
                    x0 = j
                    for share, col in ((p[i, j, 1], PAIR_A), (p[i, j, 0], PAIR_B), (p[i, j, 2], FOLD_COLOR)):
                        if share > 0.005:
                            ax.add_patch(Rectangle((x0, 12 - i), share * 0.94, 0.94, color=col, linewidth=0))
                            x0 += share * 0.94
            w = combos_per[..., None] * p
            tot = w.sum((0, 1)) / combos_per.sum()
            ax.set_xlim(0, 13)
            ax.set_ylim(0, 13)
            ax.set_aspect('equal')
            ax.set_xticks(np.arange(13) + 0.47, list(RANKS13), fontsize=6, color=INK2)
            ax.set_yticks(np.arange(13) + 0.47, list(RANKS13[::-1]), fontsize=6, color=INK2)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f'{label(key)}\nraise {100 * tot[1]:.0f}%, call {100 * tot[0]:.0f}%, fold {100 * tot[2]:.0f}%',
                         fontsize=8.5, color=INK, loc='left')
        leg = axes.flat[-1]
        leg.axis('off')
        for k, (col, name) in enumerate(((PAIR_A, 'raise'), (PAIR_B, 'call (limp)' if sit == 'sb_open' else 'call'),
                                         (FOLD_COLOR, 'fold'))):
            leg.add_patch(Rectangle((0.05, 0.78 - 0.14 * k), 0.12, 0.09, color=col, transform=leg.transAxes))
            leg.text(0.22, 0.825 - 0.14 * k, name, fontsize=9, color=INK, va='center', transform=leg.transAxes)
        leg.text(0.05, 0.25, 'Each cell = one starting hand,\nsplit by the action shares\n(mean over the 3 seeds).\n'
                 'Above the diagonal: suited;\nbelow: offsuit; diagonal: pairs.', fontsize=7.5, color=INK2,
                 transform=leg.transAxes, va='center')
        fig.suptitle(f"Pre-flop strategy: {sit_title.lower()}", fontsize=11, color=INK, x=0.01, ha='left')
        fig.tight_layout()
        _save(fig, f'preflop_{sit}', f'Pre-flop strategy of every agent ({sit_title.lower()}): share of raise, call '
              'and fold for each of the 169 starting hands, averaged over its combos and the three seeds; the title '
              'gives the shares over all 1326 combos', 'saved policies (computed; cache experiments/thesis/data/'
              'preflop.json)')
        plt.close(fig)


# ---- Leduc leak map --------------------------------------------------------- #

def _leduc_leak_rows():
    ''' Every decision point of each Leduc winner seed: its reach-weighted EV loss against CFR+ (cached). '''
    path = os.path.join(DATA, 'leduc_leaks.json')
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    import leduc as L
    from agent_loading import load_policy
    from hand_analysis import _leduc_values
    eq = L._Table(L.equilibrium(), True)
    rows = []
    for algo, seed, d, _ in _leduc_agents():
        agent = L._Table(L.policy_of(load_policy(algo, d)))
        for seat in (0, 1):
            for rank in L.RANKS:
                def walk(state, pub, my_reach, hist):
                    if state.is_terminal():
                        return
                    if state.round == 1 and pub is None:
                        for pb in L.RANKS:
                            walk(state, pb, my_reach, hist)
                        return
                    legal = state.legal_actions()
                    if state.pointer != seat:
                        for a in legal:
                            walk(state.child(a), pub, my_reach, hist + [a])
                        return
                    p = agent(state, rank, pub if state.round == 1 else None)
                    try:
                        values, norm = _leduc_values(L, hist, seat, rank, pub if state.round == 1 else None,
                                                     agent, eq, with_reach=True)
                    except ZeroDivisionError:
                        norm = 0.0
                    if norm > 0 and my_reach > 0:
                        q = np.array([values[a] for a in legal])
                        pp = np.array([p[a] for a in legal])
                        rows.append({'algo': algo, 'seed': seed, 'seat': seat, 'round': state.round + 1,
                                     'rank': L.RANK_NAMES[rank],
                                     'public': None if state.round == 0 else L.RANK_NAMES[pub],
                                     'facing': bool(state.raised[state.pointer] < max(state.raised)),
                                     'reach': 0.5 * (1 / 3) * my_reach * norm,   # per deal: seat, own card
                                     'ev_loss': float(q.max() - q @ pp),
                                     'action': L.ACTIONS[legal[int(pp.argmax())]],
                                     'best': L.ACTIONS[legal[int(q.argmax())]]})
                    for a in legal:
                        if p[a] > 0:
                            walk(state.child(a), pub, my_reach * p[a], hist + [a])
                walk(L.LeducState(), None, 1.0, [])
        print(f'  leduc leaks: {algo} seed {seed}')
    os.makedirs(DATA, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f)
    return rows


def figure_leduc_leaks(plt):
    from matplotlib.colors import LinearSegmentedColormap
    rows = _leduc_leak_rows()
    # check: the reach-weighted sum is the expected EV loss per deal; compare with the 400 sampled deals
    for k in ALGO4:
        exp = sum(r['reach'] * r['ev_loss'] for r in rows if r['algo'] == k and r['seed'] == 1042)
        dec = _leduc_decisions().get(k)
        if dec:   # per deal over all 400 deals (some have no agent decision: the opponent folded first)
            emp = sum(d['ev_loss'] for d in dec) / 400
            print(f'  leduc leaks check {k}: expected {exp:.3f} BB/deal, 400 sampled deals {emp:.3f}')
    cols = [('1', 'J'), ('1', 'Q'), ('1', 'K'), ('2', 'J'), ('2', 'Q'), ('2', 'K'), ('2', 'pair')]
    col_names = ['J', 'Q', 'K', 'J high', 'Q high', 'K high', 'pair']
    situations = [(False, 'no bet to call'), (True, 'facing a bet\n(round 1: incl. the blind)')]

    def cell_of(r):
        if r['round'] == 1:
            return ('1', r['rank'])
        return ('2', 'pair' if r['rank'] == r['public'] else r['rank'])
    grids, notes = {}, {}
    for algo in ALGO4:
        mine = [r for r in rows if r['algo'] == algo]
        n_seeds = len({r['seed'] for r in mine})
        g = np.zeros((2, len(cols)))
        mistakes = {}
        for r in mine:
            i = int(r['facing'])
            j = cols.index(cell_of(r))
            v = 1000 * r['reach'] * r['ev_loss'] / n_seeds
            g[i, j] += v
            if r['action'] != r['best']:
                mistakes.setdefault((i, j), {}).setdefault(f"{r['action']} instead of {r['best']}", 0.0)
                mistakes[(i, j)][f"{r['action']} instead of {r['best']}"] += v
        grids[algo] = g
        notes[algo] = {c: max(m, key=m.get) for c, m in mistakes.items()}
    vmax = max(g.max() for g in grids.values())
    cmap = LinearSegmentedColormap.from_list('violet', [SURFACE, '#c9c2f2', PAIR_A])
    fig, axes = plt.subplots(len(ALGO4), 1, figsize=(9, 8.6))
    order = sorted(ALGO4, key=lambda a: grids[a].sum())
    for ax, algo in zip(axes, order):
        g = grids[algo]
        ax.imshow(g, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
        for i in range(2):
            for j in range(len(cols)):
                v = g[i, j]
                dark = v > 0.55 * vmax
                txt = f'{v:.0f}'
                top3 = sorted(np.ndindex(g.shape), key=lambda c: -g[c])[:3]
                if (i, j) in top3 and (i, j) in notes[algo] and v > 0.05 * vmax:
                    txt += '\n' + notes[algo][(i, j)]
                ax.text(j, i, txt, ha='center', va='center', fontsize=6.8 if '\n' in txt else 8,
                        color=SURFACE if dark else INK, linespacing=1.1)
        ax.set_xticks(range(len(cols)), col_names, fontsize=8, color=INK2)
        ax.set_yticks(range(2), [s for _, s in situations], fontsize=8, color=INK2)
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, 2), minor=True)
        ax.grid(which='minor', color=SURFACE, linewidth=2)
        ax.axvline(2.5, color=INK2, linewidth=1)
        ax.tick_params(which='both', length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f'{LABEL[algo]}: {g.sum():.0f} mbb lost per deal in total', fontsize=9, color=INK, loc='left')
    axes[-1].set_xlabel('round 1 (own card)                                         '
                        'round 2 (own card against the board card)', fontsize=8, color=INK2)
    fig.suptitle('Leduc: where each algorithm loses against CFR+ (mbb per deal, reach-weighted; '
                 'text = its costliest wrong action)', fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    _save(fig, 'leduc_leaks', 'Leduc: expected EV loss per deal against the CFR+ equilibrium, split by round, own '
          'hand and whether the agent faces a bet (exact over every decision point, weighted by how often it is '
          'reached; mean over the three seeds); the costliest cells name the dominant wrong action. Round 1 '
          '"no bet to call" is the big blind after a small-blind call, which the CFR+ small blind almost never makes '
          '(reached in about 0.002% of deals), hence the zeros',
          'saved policies (computed; cache experiments/thesis/data/leduc_leaks.json)')
    plt.close(fig)


# --------------------------------------------------------------------------- #

def write_index():
    lines = ['# Thesis tables and figures', '',
             'Generated by `python thesis_assets.py` from finished results; rerun it after any result changes.',
             'Units: mbb/hand (milli big blinds per hand) unless stated. LaTeX tables need `\\usepackage{booktabs}`;',
             'include them with `\\input{tables/<name>.tex}` inside a `table` environment (each file is a bare `tabular`',
             'whose first line is a `%` comment with its title).', '']
    for section in ('Tables', 'Figures'):
        lines += [f'## {section}', '', '| file | caption | source |', '|---|---|---|']
        lines += [f'| `{f}` | {c} | `{s}` |' for sec, f, c, s in INDEX if sec == section]
        lines.append('')
    with open(os.path.join(OUT, 'README.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'DejaVu Sans'
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    copy_existing()
    table_overview()
    table_ablations()
    table_leduc_mistakes()
    figure_h2h_matrix(plt)
    figure_leduc_matrix(plt)
    figure_ranking(plt)
    figure_measures(plt)
    figure_ablations(plt)
    for f in (figure_rank_slope, figure_opponent_profile, figure_cost, figure_style_map, figure_style_training,
              figure_endings, figure_leduc_losses, figure_seeds, figure_importance, figure_position, figure_win_rate,
              figure_measures_games, figure_fold_discipline, figure_nfsp_divergence, figure_selection,
              figure_preflop, figure_leduc_leaks):
        f(plt)
    write_index()
    print(f'{len(INDEX)} files indexed in {os.path.relpath(OUT, REPO)}/README.md')


if __name__ == '__main__':
    main()
