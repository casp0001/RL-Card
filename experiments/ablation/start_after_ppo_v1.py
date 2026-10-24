''' Wait until the ppo_v1 tuning study is finished, then run the self-play
ablation on its best trial. Started detached; output in
experiments/ablation/ppo_v1_run.log. Safe to start again after a reboot
(the ablation skips finished runs). '''
import os
import sys
import time
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)
import tune  # noqa: E402

TARGET = 40
while True:
    study = tune.load_study('ppo_v1')
    finished, running = tune.count_finished(study), tune.count_running(study)
    print(time.strftime('%H:%M'), f'ppo_v1: {finished}/{TARGET} finished, {running} running', flush=True)
    if finished >= TARGET and running == 0:
        break
    time.sleep(300)
subprocess.run([sys.executable, '-u', 'self_play_ablation.py', 'run', '--study', 'ppo_v1',
                '--seeds', '2', '--workers', '4', '--threads', '3'], check=False)
