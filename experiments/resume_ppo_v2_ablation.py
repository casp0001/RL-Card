''' Resume the ppo_v2 self-play ablation (4 workers) once the finalists
pipeline is done and no ablation run is still training (a restarted
scheduler would wipe a run it thinks is unfinished). Started detached; log in
experiments/resume_ppo_v2_ablation.log. '''
import os
import sys
import time
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
PIPELINE_LOG = os.path.join('experiments', 'run_ppo_finalists_and_ablation.log')


def ablation_runs_alive():
    out = subprocess.run(['powershell', '-NoProfile', '-Command',
                          "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                          "? { $_.CommandLine -like '*self_play_ablation.py*train*' }).Count"],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def pipeline_done():
    return os.path.isfile(PIPELINE_LOG) and 'all done' in open(PIPELINE_LOG, encoding='utf-8').read()


NOW = '--now' in sys.argv          # do not wait for the finalists pipeline
WORKERS = sys.argv[sys.argv.index('--workers') + 1] if '--workers' in sys.argv else '4'
while not ((NOW or pipeline_done()) and ablation_runs_alive() == 0):
    print(time.strftime('%H:%M'), f'waiting: finalists pipeline done={pipeline_done()}, '
          f'ablation runs training={ablation_runs_alive()}', flush=True)
    time.sleep(120)
print(time.strftime('%H:%M'), 'resuming the ablation', flush=True)
subprocess.run([sys.executable, '-u', 'self_play_ablation.py', 'run', '--study', 'ppo_v2', '--trial', '35',
                '--seeds', '4', '--variants', 'latest,warmup_honest,warmup_mix_pool', '--workers', WORKERS,
                '--threads', '2' if NOW else '3'], check=False)
print(time.strftime('%H:%M'), 'ablation done', flush=True)
