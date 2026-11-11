''' The unattended pipeline of 2026-11-06, one stage after the other:

  1. extended limit hold'em runs of the Deep CFR and NFSP winners
     (deepcfr_v1 trial 27 to 500 iterations, nfsp_v3 trial 33 to 8M episodes,
     3 seeds each), with the held-out evaluation
  2. Leduc: for Deep CFR, PPO, NFSP and DQN in turn, a tuning study (4 workers)
     scored by exact exploitability, then its finalists (top 3 x 3 seeds)
  3. the Leduc results table (experiments/leduc_results.md)

Started detached; resumable: start it again after a restart and every stage
continues where it stopped (finished trials and runs are kept).

    python experiments/run_pipeline.py            (log: experiments/pipeline.log)
'''
import os
import sys
import json
import time
import statistics
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
PY = sys.executable
LOGS = os.path.join('experiments', 'tune', '_logs')
WORKERS, THREADS = 4, 3

# (study, algo, budget, extra tune.py run arguments)
LEDUC = [
    ('leduc_deepcfr_v1', 'deepcfr', 300, ['--space', 'experiments/tune/leduc_deepcfr_space.json']),
    ('leduc_ppo_v1', 'ppo', 1_000_000, []),
    ('leduc_nfsp_v1', 'nfsp', 2_000_000, ['--space', 'experiments/tune/nfsp_v3_space.json', '--startup_trials', '10']),
    ('leduc_dqn_v1', 'dqn', 1_000_000, ['--startup_trials', '10']),
]
TRIALS, EVALS = 30, 20


def log(msg):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {msg}', flush=True)


def parallel(cmds, name):
    ''' Run the commands at once (output to _logs/<name>_<i>.out) and wait. '''
    procs = []
    for i, cmd in enumerate(cmds, 1):
        out = open(os.path.join(LOGS, f'{name}_{i}.out'), 'a', encoding='utf-8')
        procs.append((subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=REPO), out))
        time.sleep(5)       # the first worker of a new study writes its config
    codes = []
    for proc, out in procs:
        codes.append(proc.wait())
        out.close()
    if any(codes):
        log(f'{name}: exit codes {codes} (see {LOGS}/{name}_*.out)')
    return codes


def leduc_table():
    rows = ['| algorithm | study | trial | seeds | exploitability (mbb/hand, lower = better) | vs equilibrium | '
            'vs random | winner |', '|---|---|---|---|---|---|---|---|']
    for study, algo, _, _ in LEDUC:
        path = os.path.join('experiments', 'tune', study, 'finalists', 'finalists.json')
        if not os.path.isfile(path):
            rows.append(f'| {algo} | {study} | - | - | not run | | | |')
            continue
        with open(path, encoding='utf-8') as f:
            record = json.load(f)
        win = record.get('winner', {}).get('trial')
        for trial in dict.fromkeys(r['trial'] for r in record['runs']):
            runs = [r for r in record['runs'] if r['trial'] == trial and 'failed' not in r]
            failed = sum(1 for r in record['runs'] if r['trial'] == trial and 'failed' in r)

            def agg(xs):
                if not xs:
                    return '-'
                return f'{statistics.fmean(xs):.0f}' + (f' (sd {statistics.stdev(xs):.0f})' if len(xs) > 1 else '')
            rows.append(f'| {algo} | {study} | {trial} | {len(runs)}' + (f' ({failed} failed)' if failed else '')
                        + f' | {agg([-r["final_min"] for r in runs])} | {agg([r["final_mean"] for r in runs])} | '
                        f'{agg([r["members"]["random"]["mean"] for r in runs])} | {"yes" if trial == win else ""} |')
    from leduc import equilibrium, exploitability, uniform_policy
    eq = exploitability(equilibrium(), True) * 1000
    uniform = exploitability(uniform_policy) * 1000
    text = ("Leduc hold'em, exact evaluation of the finalists (top 3 trials x 3 seeds of each study).\n"
            f'Reference: tabular CFR+ (1000 iterations) has exploitability {eq:.2f} mbb/hand, '
            f'the uniform random policy {uniform:.0f}.\n\n' + '\n'.join(rows) + '\n')
    with open(os.path.join('experiments', 'leduc_results.md'), 'w', encoding='utf-8') as f:
        f.write(text)
    log('Leduc table written to experiments/leduc_results.md\n' + text)


def main():
    os.makedirs(LOGS, exist_ok=True)
    log('stage 1: extended deepcfr_v1 (500 iterations) and nfsp_v3 (8M episodes)')
    code = subprocess.call([PY, '-u', 'experiments/run_extended.py', '--run', 'deepcfr_v1:500',
                            '--run', 'nfsp_v3:8000000', '--processes', str(WORKERS), '--threads', str(THREADS)],
                           cwd=REPO)
    log(f'stage 1 finished (exit {code})')

    for study, algo, budget, extra in LEDUC:
        log(f'stage 2: {study} tuning ({TRIALS} trials, budget {budget})')
        cmd = [PY, '-u', 'tune.py', 'run', '--algo', algo, '--study', study, '--trials', str(TRIALS),
               '--budget', str(budget), '--evals', str(EVALS), '--threads', str(THREADS),
               '--set', 'game=leduc'] + extra
        parallel([cmd] * WORKERS, f'{study}_w')
        log(f'stage 2: {study} finalists')
        cmd = [PY, '-u', 'tune.py', 'finalists', '--study', study, '--top', '3', '--seeds', '3',
               '--threads', str(THREADS)]
        parallel([cmd] * WORKERS, f'{study}_finalists_p')
        log(f'{study} done')

    leduc_table()
    log('pipeline done')


if __name__ == '__main__':
    main()
