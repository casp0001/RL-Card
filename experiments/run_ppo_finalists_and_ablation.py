''' Finalists of ppo_v1 and ppo_v2 (top 3 x 3 seeds, 2 processes each), the
self-play ablation on ppo_v2's trial 35 alongside them (3 workers), then the
held-out comparison of all finalists. Started detached; log in
experiments/run_ppo_finalists_and_ablation.log. Every step is resumable, so
after a restart simply start this script again. '''
import os
import sys
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
PY = sys.executable
LOGS = os.path.join('experiments', 'tune', '_logs')
os.makedirs(LOGS, exist_ok=True)


def start(args, log):
    return subprocess.Popen([PY, '-u'] + args, stdout=open(os.path.join(LOGS, log), 'a', encoding='utf-8'),
                            stderr=subprocess.STDOUT)


procs = []
for study in ('ppo_v1', 'ppo_v2'):
    for i in (1, 2):
        procs.append(start(['tune.py', 'finalists', '--study', study, '--top', '3', '--seeds', '3',
                            '--final_deals', '5000', '--threads', '2'], f'{study}_finalists_p{i}.out'))
ablation = start(['self_play_ablation.py', 'run', '--study', 'ppo_v2', '--trial', '35', '--seeds', '4',
                  '--variants', 'latest,warmup_honest,warmup_mix_pool', '--workers', '3', '--threads', '2'],
                 'ppo_v2_ablation.out')
print('started finalists (4 processes) and the ablation', flush=True)
for p in procs:
    p.wait()
print('finalists done; held-out evaluation', flush=True)
subprocess.run([PY, '-u', 'heldout_eval.py', 'finalists', '--study', 'ppo_v1', '--study', 'ppo_v2',
                '--workers', '4', '--threads', '2'], check=False)
ablation.wait()
print('all done', flush=True)
