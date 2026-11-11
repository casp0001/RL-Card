''' The final stage, one step after the other:

  1. extended PPO: ppo_v2's winner (trial 35) trained to 4M episodes, 3 seeds
     in parallel, with the held-out evaluation (experiments/run_extended.py)
  2. the manifest of the final evaluation: the winners of dqn_v1, ppo_v2,
     nfsp_v3 and deepcfr_v1, plus the extended runs as ppo_ext, nfsp_ext and
     deepcfr_ext (18 agents)
  3. the final evaluation (final_evaluation.py run): head-to-head round robin
     with match details, style matches, panel and LBR; tables in experiments/final/
  4. final_report.py all: learning-curve figures, training cost table, exact
     Leduc round robin, tuning tables (search spaces, winners, parameter
     importance), the NFSP divergence table, example hands against tag (and
     Leduc hands against CFR+) with the EV loss of every decision, the hand
     viewer experiments/final/hands/hands.html and
     provenance.json (git commit, versions, settings); provenance_launch.json
     records the code state at launch

It refuses to start until both ablation reports of experiments/run_ablations.py
exist (they decide whether an algorithm is retrained first).

Started detached; resumable (start it again after a restart: finished runs
and evaluation tasks are kept).

    python experiments/run_final.py            (log: experiments/final.log)
'''
import os
import sys
import time
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
PY = sys.executable

STUDIES = ['dqn=dqn_v1', 'ppo=ppo_v2', 'nfsp=nfsp_v3', 'deepcfr=deepcfr_v1']
EXTENDED = ['ppo=ppo_v2:4000000', 'nfsp=nfsp_v3:8000000', 'deepcfr=deepcfr_v1:500']


def log(msg):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {msg}', flush=True)


def step(name, cmd):
    log(name)
    code = subprocess.call(cmd, cwd=REPO)
    log(f'{name}: exit {code}')
    if code:
        raise SystemExit(code)


ABLATION_REPORTS = ['experiments/ablation_ppo_free_fold.md', 'experiments/ablation_nfsp_batchnorm.md']


def main():
    # the ablations (experiments/run_ablations.py) decide whether anything is retrained first
    missing = [p for p in ABLATION_REPORTS if not os.path.isfile(p)]
    if missing and '--skip_ablation_check' not in sys.argv:
        raise SystemExit(f'ablation report(s) missing: {missing}. Run experiments/run_ablations.py first and '
                         'decide whether anything must be retrained (--skip_ablation_check to start anyway).')
    # the code state at launch (the run itself does not change it)
    step('stage 0: provenance at launch',
         [PY, '-u', 'final_report.py', 'provenance', '--name', 'provenance_launch.json'])
    step('stage 1: extended ppo_v2 (4M episodes)',
         [PY, '-u', 'experiments/run_extended.py', '--run', 'ppo_v2:4000000', '--processes', '3', '--threads', '4'])
    manifest = [PY, '-u', 'final_evaluation.py', 'manifest']
    for s in STUDIES:
        manifest += ['--study', s]
    for s in EXTENDED:
        manifest += ['--extended', s]
    step('stage 2: manifest', manifest)
    step('stage 3: final evaluation', [PY, '-u', 'final_evaluation.py', 'run'])
    step('stage 4: learning curves, training cost, exact Leduc head-to-head, provenance',
         [PY, '-u', 'final_report.py', 'all'])
    log('all done; tables in experiments/final/ (summary.md, cost.md, leduc_h2h.md), figures in '
        'experiments/final/figures/')


if __name__ == '__main__':
    main()
