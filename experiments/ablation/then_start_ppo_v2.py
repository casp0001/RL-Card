''' Wait for the ppo_v1 self-play ablation to finish, pick the best variant
and start the ppo_v2 tuning study with it (4 workers). Started detached;
output in experiments/ablation/then_start_ppo_v2.log.

Selection rule (fixed before the results were in): rank the variants by mean
LBR over seeds (lower = less exploitable) and by mean held-out panel min
(higher = better); the best sum of ranks wins, LBR breaks ties. If plain
self-play ('latest') wins, ppo_v2 would only repeat ppo_v1, so nothing is
started.

The choice is written to experiments/ablation/ppo_v1/chosen_variant.json and
the exact start command to experiments/tune/run_ppo_v2.ps1 (run that file to
resume ppo_v2 after a restart).
'''
import os
import sys
import json
import glob
import time
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)
import tune  # noqa: E402
from self_play_ablation import VARIANTS, study_root  # noqa: E402

RUNS = len(VARIANTS) * 2
WARMUP_FRAC = 0.2
root = study_root('ppo_v1')

while len(glob.glob(os.path.join(root, '*', '*', 'result.json'))) < RUNS:
    print(time.strftime('%H:%M'), f'{len(glob.glob(os.path.join(root, "*", "*", "result.json")))}/{RUNS} '
          f'ablation runs done', flush=True)
    time.sleep(120)

scores = {}
for v in VARIANTS:
    results = [json.load(open(p, encoding='utf-8')) for p in glob.glob(os.path.join(root, v, '*', 'result.json'))]
    if results:
        scores[v] = {'lbr': sum(r['lbr']['mean'] for r in results) / len(results),
                     'panel_min': sum(r['panel']['min'] for r in results) / len(results), 'seeds': len(results)}
by_lbr = sorted(scores, key=lambda v: scores[v]['lbr'])
by_panel = sorted(scores, key=lambda v: -scores[v]['panel_min'])
for v in scores:
    scores[v]['rank_sum'] = by_lbr.index(v) + by_panel.index(v)
best = min(scores, key=lambda v: (scores[v]['rank_sum'], scores[v]['lbr']))
print(json.dumps(scores, indent=1))
print('best variant:', best, flush=True)

budget = tune.load_config('ppo_v1')['budget']
sets = {k: (int(round(WARMUP_FRAC * budget)) if val == 'W' else val) for k, val in VARIANTS[best][1].items()}
with open(os.path.join(root, 'chosen_variant.json'), 'w', encoding='utf-8') as f:
    json.dump({'variant': best, 'settings': sets, 'scores': scores,
               'rule': 'best sum of ranks by mean LBR (low) and mean held-out panel min (high); LBR breaks ties',
               'chosen': time.strftime('%Y-%m-%d %H:%M')}, f, indent=1)
if best == 'latest':
    print('plain self-play won: ppo_v2 would repeat ppo_v1, not starting it', flush=True)
    sys.exit(0)

set_args = ' '.join(f'--set {k}={v}' for k, v in sets.items())
script = os.path.join('experiments', 'tune', 'run_ppo_v2.ps1')
with open(script, 'w', encoding='utf-8') as f:
    f.write(f"""# Start (or, after a restart / shutdown, resume) the PPO tuning study ppo_v2 with 4 workers:
# the ppo_v1 search space with the self-play variant chosen by the ablation ({best}).
# Run from the repository root:   powershell -ExecutionPolicy Bypass -File experiments\\tune\\run_ppo_v2.ps1
# Interrupted trials are detected and re-run automatically. Logs: experiments\\tune\\_logs\\
$study = 'ppo_v2'
$workers = 4
$cmd = "-u tune.py run --algo ppo --study $study --trials 50 --threads 3 {set_args}"
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
print(f'wrote {script}; starting ppo_v2 with {set_args}', flush=True)
subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', script], check=False)
print('ppo_v2 started', flush=True)
