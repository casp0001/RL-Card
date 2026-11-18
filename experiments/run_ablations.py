''' Several ablations of experiments/run_ablation.py side by side, each in its
own process (log: experiments/ablation_<name>.log).

Default (no names): the two ablations of the code review, the PPO free-fold
ablation and the NFSP BatchNorm ablation. Their results decide whether any
algorithm has to be retrained, so they run BEFORE the final pipeline
(experiments/run_final.py refuses to start until both reports exist).
3 PPO runs (1M episodes) and 3 NFSP runs (2M episodes) at once, 2 threads each;
about 2.5-3 h in total.

    python experiments/run_ablations.py                              (log: experiments/ablations.log)
    python experiments/run_ablations.py deepcfr_sgd nfsp_literature  (the configuration follow-ups)
Resumable: start it again after a restart.
'''
import os
import sys
import time
import argparse
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
DEFAULT = ['ppo_free_fold', 'nfsp_batchnorm']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('names', nargs='*', default=DEFAULT, help='ablations of run_ablation.py (default: %(default)s)')
    p.add_argument('--threads', type=int, default=2, help='torch threads per training run / measurement')
    p.add_argument('--processes', type=int, default=3, help='measurements at once per ablation')
    a = p.parse_args()
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  starting {", ".join(a.names)}', flush=True)
    procs = []
    for name in a.names:
        log = open(os.path.join('experiments', f'ablation_{name}.log'), 'a', encoding='utf-8')
        procs.append((name, log, subprocess.Popen(
            [sys.executable, '-u', 'experiments/run_ablation.py', name, '--threads', str(a.threads),
             '--processes', str(a.processes)],
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
