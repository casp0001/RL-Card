''' Unattended follow-up of the ppo_v2 self-play ablation (started detached;
log in experiments/after_ablation.log).

1. Wait until the ablation (12 runs) and the finalists pipeline are done.
2. Decide, with rules fixed before the results are in:
     best variant      best sum of ranks by mean LBR (low) and mean held-out
                       panel min (high) over seeds; LBR breaks ties
     mix_pool clearly  warmup_mix_pool has the best mean LBR *and* the best
     better            mean panel min, and its LBR lead over each other variant
                       exceeds one standard error of the difference (seeds)
3. If warmup_mix_pool is clearly better: run ppo_v3 (PPO space, that variant,
   50 trials); when ppo_v3 has finished, run dqn_v1.
   Otherwise: run dqn_v1 right away.
   dqn_v1 uses the best variant's self-play setup (no warm-up if plain
   self-play wins), 50 trials, the widened DQN space from tune_spaces.py.

The decision is written to experiments/ablation/ppo_v2/decision.json, each
study's exact start command to experiments/tune/run_<study>.ps1 (run that
file to resume the study after a restart). Restarting this script is safe:
it skips studies that are already finished and resumes unfinished ones.
'''
import os
import sys
import glob
import json
import math
import time
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
import tune  # noqa: E402

ROOT = os.path.join('experiments', 'ablation', 'ppo_v2')
VARIANTS = ('latest', 'warmup_honest', 'warmup_mix_pool')
SEEDS = 4
WARMUP = 200_000                  # 20 % of the 1M-episode budgets
SETS = {'latest': {},
        'warmup_honest': {'warmup_episodes': WARMUP, 'warmup_bot': 'honest'},
        'warmup_mix_pool': {'warmup_episodes': WARMUP, 'warmup_bot': 'equity_mix', 'opponent_pool': 10}}
TRIALS = 50


def log(msg):
    print(time.strftime('%Y-%m-%d %H:%M'), msg, flush=True)


def text(path):
    return open(path, encoding='utf-8').read() if os.path.isfile(path) else ''


def ablation_done():
    return len(glob.glob(os.path.join(ROOT, '*', '*', 'result.json'))) >= len(VARIANTS) * SEEDS and \
        'ablation done' in text(os.path.join('experiments', 'resume_ppo_v2_ablation.log'))


def finalists_done():
    return 'all done' in text(os.path.join('experiments', 'run_ppo_finalists_and_ablation.log'))


def study_state(study):
    if not os.path.isfile(tune.config_path(study)):
        return None, 0, 0
    s = tune.load_study(study)
    return s, tune.count_finished(s), tune.count_running(s)


def mean_sd(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def decide():
    stats = {}
    for v in VARIANTS:
        rs = [json.load(open(p, encoding='utf-8')) for p in glob.glob(os.path.join(ROOT, v, '*', 'result.json'))]
        lbr, panel = mean_sd([r['lbr']['mean'] for r in rs]), mean_sd([r['panel']['min'] for r in rs])
        stats[v] = {'seeds': len(rs), 'lbr_mean': lbr[0], 'lbr_sd': lbr[1],
                    'panel_min_mean': panel[0], 'panel_min_sd': panel[1]}
    by_lbr = sorted(stats, key=lambda v: stats[v]['lbr_mean'])
    by_panel = sorted(stats, key=lambda v: -stats[v]['panel_min_mean'])
    for v in stats:
        stats[v]['rank_sum'] = by_lbr.index(v) + by_panel.index(v)
    best = min(stats, key=lambda v: (stats[v]['rank_sum'], stats[v]['lbr_mean']))
    mp = stats['warmup_mix_pool']
    margins = {}
    for v in VARIANTS:
        if v == 'warmup_mix_pool':
            continue
        o = stats[v]
        se = math.sqrt(o['lbr_sd'] ** 2 / o['seeds'] + mp['lbr_sd'] ** 2 / mp['seeds'])
        margins[v] = {'lbr_lead': o['lbr_mean'] - mp['lbr_mean'], 'se_of_difference': se}
    clearly = (by_lbr[0] == 'warmup_mix_pool' and by_panel[0] == 'warmup_mix_pool'
               and all(m['lbr_lead'] > m['se_of_difference'] for m in margins.values()))
    return {'stats': stats, 'best_variant': best, 'mix_pool_clearly_better': clearly, 'mix_pool_margins': margins,
            'rules': {'best': 'best sum of ranks by mean LBR (low) and mean held-out panel min (high); LBR breaks ties',
                      'clearly': 'warmup_mix_pool best on mean LBR and mean panel min, LBR lead over each other '
                                 'variant > 1 standard error of the difference'},
            'decided': time.strftime('%Y-%m-%d %H:%M')}


def start_study(study, algo, sets, extra=''):
    set_args = ' '.join(f'--set {k}={v}' for k, v in sets.items())
    script = os.path.join('experiments', 'tune', f'run_{study}.ps1')
    with open(script, 'w', encoding='utf-8') as f:
        f.write(f"""# Start (or, after a restart / shutdown, resume) the {algo.upper()} tuning study {study} with 4 workers.
# Written by experiments/after_ablation.py; self-play setup: {sets or 'plain self-play'}
# Run from the repository root:   powershell -ExecutionPolicy Bypass -File experiments\\tune\\run_{study}.ps1
# Interrupted trials are detected and re-run automatically. Logs: experiments\\tune\\_logs\\
$study = '{study}'
$workers = 4
$cmd = "-u tune.py run --algo {algo} --study $study --trials {TRIALS} --threads 3 {extra} {set_args}"
New-Item -ItemType Directory -Force experiments\\tune\\_logs | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
foreach ($i in 1..$workers) {{
    Start-Process python -ArgumentList $cmd -WindowStyle Hidden `
        -RedirectStandardOutput "experiments\\tune\\_logs\\${{study}}_${{stamp}}_w$i.out" `
        -RedirectStandardError "experiments\\tune\\_logs\\${{study}}_${{stamp}}_w$i.err"
    Start-Sleep 3
}}
Write-Host "started $workers workers for $study; progress: python tune.py status --study $study"
""")
    log(f'starting {study}: {algo}, {TRIALS} trials, {set_args or "plain self-play"} {extra}')
    subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', script], check=False)


def workers_alive(study):
    ''' tune.py worker processes of a study (after a reboot there are none,
    although its interrupted trials still look RUNNING until a worker starts). '''
    out = subprocess.run(['powershell', '-NoProfile', '-Command',
                          "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { $_.CommandLine -like "
                          f"'*tune.py run*--study {study} *' }}).Count"], capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def run_study_to_end(study, algo, sets, extra=''):
    ''' Start (or resume) a study and wait until it has finished. '''
    _, finished, running = study_state(study)
    if finished >= TRIALS and running == 0:
        log(f'{study} already finished')
        return
    if workers_alive(study) == 0:
        start_study(study, algo, sets, extra)
    while True:
        time.sleep(300)
        _, finished, running = study_state(study)
        log(f'{study}: {finished}/{TRIALS} finished, {running} running')
        if finished >= TRIALS and running == 0:
            log(f'{study} finished')
            return
        if workers_alive(study) == 0:
            # every worker exited (e.g. a trial failed late, leaving the study one
            # trial short, see tune.py): start them again to fill the gap
            log(f'{study}: no workers left at {finished}/{TRIALS}; starting them again')
            start_study(study, algo, sets, extra)


while not (ablation_done() and finalists_done()):
    log(f'waiting: ablation done={ablation_done()}, finalists pipeline done={finalists_done()}')
    time.sleep(300)

decision_path = os.path.join(ROOT, 'decision.json')
if os.path.isfile(decision_path):
    decision = json.load(open(decision_path, encoding='utf-8'))
    log('using the earlier decision')
else:
    decision = decide()
    with open(decision_path, 'w', encoding='utf-8') as f:
        json.dump(decision, f, indent=1)
log(json.dumps(decision, indent=1))
log(f'best variant: {decision["best_variant"]}; warmup_mix_pool clearly better: {decision["mix_pool_clearly_better"]}')

if decision['mix_pool_clearly_better']:
    run_study_to_end('ppo_v3', 'ppo', SETS['warmup_mix_pool'])
# more random startup trials for DQN's larger, wider space (8 tuned parameters)
run_study_to_end('dqn_v1', 'dqn', SETS[decision['best_variant']], extra='--startup_trials 15')
log('all done')
