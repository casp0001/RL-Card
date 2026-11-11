''' Both ablations of the code review, side by side (experiments/run_ablation.py):
the PPO free-fold ablation and the NFSP BatchNorm ablation. Their results
decide whether any algorithm has to be retrained, so they run BEFORE the final
pipeline (experiments/run_final.py refuses to start until both reports exist).

Training: 3 PPO runs (1M episodes, ~1-1.5 h) and 3 NFSP runs (2M episodes,
~2 h) at once, 2 threads each; then each ablation measures its arms and plays
the head-to-head matches. About 2.5-3 h in total.

    python experiments/run_ablations.py         (log: experiments/ablations.log)
Resumable: start it again after a restart.
'''
import os
import sys
import time
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
JOBS = [('ppo_free_fold', '2'), ('nfsp_batchnorm', '2')]


def main():
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  starting {", ".join(j for j, _ in JOBS)}', flush=True)
    procs = []
    for name, threads in JOBS:
        log = open(os.path.join('experiments', f'ablation_{name}.log'), 'a', encoding='utf-8')
        procs.append((name, log, subprocess.Popen(
            [sys.executable, '-u', 'experiments/run_ablation.py', name, '--threads', threads, '--processes', '3'],
            stdout=log, stderr=subprocess.STDOUT, cwd=REPO)))
    failed = []
    for name, log, p in procs:
        code = p.wait()
        log.close()
        print(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {name}: '
              + ('done -> experiments/ablation_%s.md' % name if code == 0
                 else f'FAILED (exit {code}, see experiments/ablation_{name}.log)'), flush=True)
        failed += [name] if code else []
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
