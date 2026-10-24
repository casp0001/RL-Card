''' NFSP: cost pilot, then the tuning study (unattended; started detached,
log in experiments/nfsp_pipeline.log).

1. Wait until dqn_v1 has finished (no other study competes for the CPU).
2. Pilot: train short NFSP runs (PILOT_EPISODES, 4 at a time, the Honest
   warm-up scaled to 20 %) for every network size (at train_every 128) and
   every train_every (at 256x256) of the NFSP search space, and time them.
   Results: experiments/tune/nfsp_pilot/pilot.json.
3. Ranges from the costs (rules fixed in advance):
     - budget 2M episodes; 1M if the starting configuration's projected full
       trial would exceed MAX_TRIAL_HOURS
     - projected cost of a (layers, train_every) pair = base time x layer
       factor x train_every factor (factors measured in the pilot); drop the
       most expensive extreme choice until no kept pair exceeds
       MAX_TRIAL_HOURS (the start values are always kept)
   The resulting space: experiments/tune/nfsp_v1_space.json.
4. Run nfsp_v1: 50 trials, 4 workers, the self-play setup the PPO ablation
   chose (Honest warm-up, 20 % of the budget), 15 random startup trials.

Restarting this script is safe: the pilot results and the space are reused,
an unfinished study is resumed.
'''
import os
import sys
import json
import time
import ctypes
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

PILOT_EPISODES = int(os.environ.get('NFSP_PILOT_EPISODES', 100_000))     # env override for testing
PILOT_DIR = os.environ.get('NFSP_PILOT_DIR', os.path.join('experiments', 'tune', 'nfsp_pilot'))
SPACE_FILE = os.path.join('experiments', 'tune', 'nfsp_v1_space.json')
MAX_TRIAL_HOURS = 6.0          # projected full trial with 4 workers in parallel
TRIALS = 50
BASE = ('256 256', 128)


def peak_memory_gb():
    class PMC(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p          # a 64-bit pseudo-handle
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize / 1e9


def pilot_one(layers, train_every, out_dir):
    ''' (subprocess) one timed pilot run '''
    import torch
    torch.set_num_threads(3)
    import tune
    import tune_spaces
    space = tune_spaces.default_space('nfsp')
    params = tune_spaces.start_params(space)
    params.update(layers=layers, train_every=int(train_every))
    cfg = {'algo': 'nfsp', 'budget': PILOT_EPISODES, 'evals': 1, 'eval_deals': 200,
           'eval_panel': 'random,station,maniac,nit,tag,lag', 'space': space,
           'fixed': {**tune_spaces.FIXED['nfsp'], 'warmup_episodes': PILOT_EPISODES // 5, 'warmup_bot': 'honest'}}
    module, args = tune.build_args(cfg, params, out_dir, 42, save_model=False)
    results = []
    t0 = time.time()
    module.train(args, lambda episode, r: results.append(r['min']) and False)
    out = {'layers': layers, 'train_every': int(train_every), 'seconds': time.time() - t0,
           'episodes': PILOT_EPISODES, 'peak_memory_gb': peak_memory_gb(), 'panel_min': results[-1] if results else None}
    with open(os.path.join(out_dir, 'pilot_result.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f)


def run_pilot():
    import tune_spaces
    space = tune_spaces.default_space('nfsp')
    configs = [(l, BASE[1]) for l in space['layers']['choices']]
    configs += [(BASE[0], te) for te in space['train_every']['choices'] if (BASE[0], te) not in configs]
    queue, running = [], []
    for layers, te in configs:
        d = os.path.join(PILOT_DIR, f'{layers.replace(" ", "x")}_te{te}')
        if not os.path.isfile(os.path.join(d, 'pilot_result.json')):
            queue.append((layers, te, d))
    log(f'pilot: {len(configs)} configurations, {len(queue)} to run, {PILOT_EPISODES} episodes each')
    while queue or running:
        while queue and len(running) < 4:
            layers, te, d = queue.pop(0)
            os.makedirs(d, exist_ok=True)
            cmd = [sys.executable, '-u', os.path.abspath(__file__), 'pilot_one', layers, str(te), d + '/']
            running.append((subprocess.Popen(cmd, stdout=open(os.path.join(d, 'run.log'), 'w'),
                                             stderr=subprocess.STDOUT), layers, te))
        time.sleep(15)
        for item in list(running):
            if item[0].poll() is not None:
                running.remove(item)
                log(f'  pilot {item[1]} / train_every {item[2]} finished (exit {item[0].returncode})')
    results = []
    for layers, te in configs:
        p = os.path.join(PILOT_DIR, f'{layers.replace(" ", "x")}_te{te}', 'pilot_result.json')
        if os.path.isfile(p):
            results.append(json.load(open(p, encoding='utf-8')))
    with open(os.path.join(PILOT_DIR, 'pilot.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1)
    return results


def choose_space(results):
    import tune_spaces
    space = tune_spaces.default_space('nfsp')
    t = {(r['layers'], r['train_every']): r['seconds'] for r in results}
    if BASE not in t:
        raise SystemExit('the pilot of the starting configuration failed; see experiments/tune/nfsp_pilot')
    per_100k = t[BASE] / (PILOT_EPISODES / 100_000)
    budget = 2_000_000
    if per_100k * budget / 100_000 / 3600 > MAX_TRIAL_HOURS:
        budget = 1_000_000
    hours = lambda seconds: seconds / PILOT_EPISODES * budget / 3600      # noqa: E731
    f_layers = {l: t[(l, BASE[1])] / t[BASE] for l in space['layers']['choices'] if (l, BASE[1]) in t}
    f_te = {te: t[(BASE[0], te)] / t[BASE] for te in space['train_every']['choices'] if (BASE[0], te) in t}
    layers, tes = list(f_layers), list(f_te)
    predicted = lambda l, te: hours(t[BASE]) * f_layers[l] * f_te[te]     # noqa: E731
    while True:
        worst = max(((l, te) for l in layers for te in tes), key=lambda p: predicted(*p))
        if predicted(*worst) <= MAX_TRIAL_HOURS:
            break
        l, te = worst
        drop_layer = (l != BASE[0]) and (f_layers[l] >= f_te[te] or te == BASE[1])
        if drop_layer:
            layers.remove(l)
        elif te != BASE[1]:
            tes.remove(te)
        else:
            break
    space['layers']['choices'] = [c for c in space['layers']['choices'] if c in layers]
    space['train_every']['choices'] = [c for c in space['train_every']['choices'] if c in tes]
    grid = [predicted(l, te) for l in layers for te in tes]
    summary = {'budget': budget, 'base_hours_full_trial': hours(t[BASE]), 'layer_cost_factor': f_layers,
               'train_every_cost_factor': f_te, 'kept_layers': layers, 'kept_train_every': tes,
               'mean_full_trial_hours_over_kept_grid': sum(grid) / len(grid),
               'study_estimate_hours': sum(grid) / len(grid) * TRIALS * 0.5 / 4}
    return space, budget, summary


def log(msg):
    print(time.strftime('%Y-%m-%d %H:%M'), msg, flush=True)


def main():
    from study_runner import run_study_to_end, study_state, workers_alive
    while True:
        _, finished, running = study_state('dqn_v1')
        if finished >= 50 and running == 0 and workers_alive('dqn_v1') == 0:
            break
        log(f'waiting for dqn_v1: {finished}/50 finished, {running} running')
        time.sleep(600)
    os.makedirs(PILOT_DIR, exist_ok=True)
    if os.path.isfile(SPACE_FILE):
        space = json.load(open(SPACE_FILE, encoding='utf-8'))
        summary = json.load(open(os.path.join(PILOT_DIR, 'decision.json'), encoding='utf-8'))
        log('using the earlier pilot decision')
    else:
        results = run_pilot()
        for r in results:
            log(f"  {r['layers']:12s} train_every {r['train_every']:3d}: {r['seconds'] / 60:5.1f} min per "
                f"{r['episodes']} episodes, peak {r['peak_memory_gb']:.2f} GB, panel min {r['panel_min']}")
        space, budget, summary = choose_space(results)
        with open(SPACE_FILE, 'w', encoding='utf-8') as f:
            json.dump(space, f, indent=1)
        with open(os.path.join(PILOT_DIR, 'decision.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=1)
    log(json.dumps(summary, indent=1))
    budget = summary['budget']
    run_study_to_end('nfsp_v1', 'nfsp', TRIALS,
                     extra=f'--budget {budget} --space {SPACE_FILE} --startup_trials 15',
                     sets={'warmup_episodes': budget // 5, 'warmup_bot': 'honest'},
                     note=f'space from the cost pilot ({SPACE_FILE}); Honest warm-up as chosen by the PPO ablation')
    log('all done')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'pilot_one':
        pilot_one(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()
