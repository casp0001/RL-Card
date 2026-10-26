''' Stop nfsp_v1 early and continue with nfsp_v2, whose ranges are extended
where nfsp_v1's best trials hit a bound (unattended; started detached, log in
experiments/nfsp_v2_pipeline.log).

1. Wait until nfsp_v1 has STOP_AT finished trials, then stop its workers (the
   trials in progress are lost; they are marked interrupted if nfsp_v1 is
   ever resumed).
2. nfsp_v2 space = nfsp_v1's space with
     rl_learning_rate   0.02 .. 1.0     (v1: 2e-3 .. 0.3; best 0.17 - 0.27)
     anticipatory_param 0.1 .. 0.9      (v1: 0.03 .. 0.5; best 0.31 - 0.46)
     train_every        8, 16, 32, 64   (v1: 32 .. 256; best mostly 32)
     q_replay_memory    20k .. 300k     (v1: 100k .. 600k; best 100k - 170k)
   everything else unchanged; written to experiments/tune/nfsp_v2_space.json.
3. Create nfsp_v2 (same budget, warm-up, 50 trials, 15 startup trials as
   nfsp_v1), queue nfsp_v1's best completed configurations first (warm
   start), start 4 workers and wait until it has finished.
Restarting this script is safe.
'''
import os
import sys
import json
import time
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
import tune  # noqa: E402
from study_runner import log, study_state, workers_alive, start_study, run_study_to_end  # noqa: E402

STOP_AT = 35
WARM_START = 5
TRIALS = 50
V1_SPACE = os.path.join('experiments', 'tune', 'nfsp_v1_space.json')
V2_SPACE = os.path.join('experiments', 'tune', 'nfsp_v2_space.json')
EXTRA_SETS_FROM_V1 = ('warmup_episodes', 'warmup_bot')


def stop_v1():
    while True:
        _, finished, running = study_state('nfsp_v1')
        if finished >= STOP_AT or workers_alive('nfsp_v1') == 0:
            break
        log(f'nfsp_v1: {finished}/{STOP_AT} finished, {running} running')
        time.sleep(120)
    subprocess.run(['powershell', '-NoProfile', '-Command',
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { $_.CommandLine -like "
                    "'*tune.py run*--study nfsp_v1 *' } | % { Stop-Process -Id $_.ProcessId -Force }"], check=False)
    time.sleep(5)
    log(f'stopped nfsp_v1 at {study_state("nfsp_v1")[1]} finished trials')


def make_space():
    space = json.load(open(V1_SPACE, encoding='utf-8'))
    space['rl_learning_rate'].update(low=0.02, high=1.0)
    space['anticipatory_param'].update(low=0.1, high=0.9)
    space['train_every']['choices'] = [8, 16, 32, 64]
    space['q_replay_memory_size'].update(low=20_000, high=300_000)
    return space


def main():
    # the pipeline that runs nfsp_v1 would restart its workers once they are stopped
    subprocess.run(['powershell', '-NoProfile', '-Command',
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { $_.CommandLine -like "
                    "'*nfsp_pipeline.py*' } | % { Stop-Process -Id $_.ProcessId -Force }"], check=False)
    if not os.path.isfile(tune.config_path('nfsp_v2')):
        stop_v1()
    if '--stop-only' in sys.argv:
        log('nfsp_v1 stopped; not starting nfsp_v2 (--stop-only). Start it with: '
            'python experiments\\nfsp_v2_pipeline.py')
        return
    v1_cfg = tune.load_config('nfsp_v1')
    budget = v1_cfg['budget']
    sets = {k: v1_cfg['fixed'][k] for k in EXTRA_SETS_FROM_V1 if k in v1_cfg['fixed']}
    if not os.path.isfile(V2_SPACE):
        space = make_space()
        # the start values (the first queued trial) = nfsp_v1's best completed configuration
        v1 = tune.load_study('nfsp_v1')
        done = sorted((t for t in v1.trials if t.state.name == 'COMPLETE'), key=lambda t: -t.value)
        for name in space:
            if space[name].get('tune', True) and done:
                space[name]['start'] = done[0].params[name]
        with open(V2_SPACE, 'w', encoding='utf-8') as f:
            json.dump(space, f, indent=1)
    extra = f'--budget {budget} --space {V2_SPACE} --startup_trials 15'
    set_args = [a for k, v in sets.items() for a in ('--set', f'{k}={v}')]
    if not os.path.isfile(tune.config_path('nfsp_v2')):
        # create the study (queues the start values = v1's best) without running a trial ...
        subprocess.run([sys.executable, 'tune.py', 'run', '--algo', 'nfsp', '--study', 'nfsp_v2', '--trials', '0',
                        '--threads', '3'] + extra.split() + set_args, check=True)
        # ... and queue v1's next best configurations behind it (warm start)
        space = json.load(open(V2_SPACE, encoding='utf-8'))
        v1 = tune.load_study('nfsp_v1')
        v2 = tune.load_study('nfsp_v2')
        done = sorted((t for t in v1.trials if t.state.name == 'COMPLETE'), key=lambda t: -t.value)
        for t in done[1:WARM_START]:
            if not tune.check_start(space, t.params):
                v2.enqueue_trial(t.params, user_attrs={'warm_start_from': f'nfsp_v1 trial {t.number}'})
                log(f'queued nfsp_v1 trial {t.number} (score {t.value:+.0f}) in nfsp_v2')
    run_study_to_end('nfsp_v2', 'nfsp', TRIALS, extra=extra, sets=sets,
                     note=f'ranges extended from nfsp_v1 ({V2_SPACE}); warm start from nfsp_v1\'s best')
    log('all done')


if __name__ == '__main__':
    main()
