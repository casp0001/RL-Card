''' Finalists of one study (top 3 x 3 seeds, several processes), then the
held-out evaluation of its finalists and a comparison table with other
studies' finalists. Started detached; resumable (start it again after a
restart: finished runs are skipped, an unfinished one restarts).

    python experiments/run_finalists.py dqn_v1 [--processes 4] [--compare ppo_v1,ppo_v2]
'''
import os
import sys
import argparse
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
p = argparse.ArgumentParser()
p.add_argument('study')
p.add_argument('--processes', type=int, default=4)
p.add_argument('--threads', type=int, default=3)
p.add_argument('--compare', default='')
a = p.parse_args()

LOGS = os.path.join('experiments', 'tune', '_logs')
os.makedirs(LOGS, exist_ok=True)
procs = [subprocess.Popen([sys.executable, '-u', 'tune.py', 'finalists', '--study', a.study, '--top', '3',
                           '--seeds', '3', '--final_deals', '5000', '--threads', str(a.threads)],
                          stdout=open(os.path.join(LOGS, f'{a.study}_finalists_p{i}.out'), 'a', encoding='utf-8'),
                          stderr=subprocess.STDOUT)
         for i in range(1, a.processes + 1)]
print(f'started {a.processes} finalists processes for {a.study}', flush=True)
for proc in procs:
    proc.wait()
print('finalists done; held-out evaluation', flush=True)
studies = [s for s in a.compare.split(',') if s] + [a.study]
subprocess.run([sys.executable, '-u', 'heldout_eval.py', 'finalists', '--workers', str(a.processes),
                '--threads', str(a.threads)] + [x for s in studies for x in ('--study', s)], check=False)
print('all done', flush=True)
