''' Hyper-parameter tuning for the four limit hold'em agents with Optuna.

Bayesian search (TPE sampler) plus early stopping of weak trials
(successive-halving / ASHA pruner). Each trial trains one configuration
in-process through the training script's train(args, on_eval); the score is
the evaluation panel's 'min' (robustness score, see eval_panel.py), reported
after every panel evaluation so that the pruner can stop weak trials, and the
trial's value is the mean of its last 3 reports.

Everything for a study lives in experiments/tune/<study>/:
    config.json      algorithm, budget, search space, ... (fixed per study)
    journal.log      Optuna storage (several worker processes may share it)
    trials/trial_N/  each trial's training logs (panel.csv, performance.csv, ...)
    finalists/       re-runs of the best trials with new seeds
    plots/           figures written by `report`

Commands:
    python tune.py run --algo dqn --study dqn_v1 --trials 40
        (start several `run` processes on the same study to work in parallel)
    python tune.py status --study dqn_v1 [--json]
    python tune.py finalists --study dqn_v1 --top 3 --seeds 3
    python tune.py report --study dqn_v1
'''

import os
import gc
import sys
import json
import math
import time
import argparse
import warnings
import importlib
import statistics

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TUNE_ROOT = os.path.join(REPO_ROOT, 'experiments', 'tune')
sys.path.insert(0, REPO_ROOT)

import tune_spaces  # noqa: E402

# TPE's multivariate / constant_liar options are flagged experimental.
warnings.filterwarnings('ignore', message='.*experimental feature.*')

# Settings that must agree between every worker of a study.
CONFIG_KEYS = ('algo', 'budget', 'evals', 'eval_deals', 'eval_panel', 'seed', 'space',
               'fixed', 'prune', 'min_resource_frac', 'reduction_factor', 'startup_trials')


# --------------------------------------------------------------------------- #
# Study files
# --------------------------------------------------------------------------- #

def study_dir(study):
    return os.path.join(TUNE_ROOT, study)


def config_path(study):
    return os.path.join(study_dir(study), 'config.json')


def load_config(study):
    path = config_path(study)
    if not os.path.isfile(path):
        raise SystemExit(f'study {study!r} does not exist ({path} missing)')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def create_or_check_config(study, cfg):
    ''' Write the config of a new study, or check that it matches the
    existing one (all workers of a study must use the same settings).
    Returns (config, created). '''
    os.makedirs(study_dir(study), exist_ok=True)
    path = config_path(study)
    try:
        with open(path, 'x', encoding='utf-8') as f:   # 'x': fails if it exists
            json.dump(cfg, f, indent=2)
        return cfg, True
    except FileExistsError:
        pass
    for _ in range(50):   # another worker may be writing it right now
        try:
            existing = load_config(study)
            break
        except (ValueError, OSError):
            time.sleep(0.1)
    diff = [k for k in CONFIG_KEYS if existing.get(k) != cfg.get(k)]
    if diff:
        raise SystemExit(f'study {study!r} already exists with different settings: {diff}.\n'
                         f'Use a new --study name, or the same settings as {path}.')
    return existing, False


def check_start(space, params):
    ''' Problems with a starting configuration (values outside the edited
    ranges / choices); an empty list when it can be enqueued. '''
    problems = []
    for name, value in params.items():
        spec = space[name]
        if spec['type'] == 'categorical':
            if value not in spec['choices']:
                problems.append(f'{name}={value!r} not in {spec["choices"]}')
        elif not spec['low'] <= value <= spec['high']:
            problems.append(f'{name}={value} outside [{spec["low"]}, {spec["high"]}]')
    return problems


def journal_path(study):
    return os.path.join(study_dir(study), 'journal.log')


def storage(study):
    import optuna
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    path = journal_path(study)
    repair_journal(path)
    # The default symlink lock needs extra privileges on Windows.
    return optuna.storages.JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


class _FileLock(object):
    ''' Optuna's O_EXCL lock file (a lock older than 30 s is broken). '''

    def __init__(self, path):
        from optuna.storages.journal import JournalFileOpenLock
        self.lock = JournalFileOpenLock(path)

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, *exc):
        self.lock.release()


def repair_journal(path):
    ''' A power cut while a worker writes can leave a half-written last line.
    Optuna skips it when reading but appends the next record right after it,
    which makes the whole journal unreadable. Cut such a tail off (it is
    kept in journal.log.tail-<time>) under the journal's own lock. '''
    if not os.path.isfile(path):
        return
    with _FileLock(path):
        size = os.path.getsize(path)
        if size == 0:
            return
        with open(path, 'rb+') as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) == b'\n':
                return
            f.seek(max(0, size - (1 << 20)))
            tail = f.read()
            cut = tail.rfind(b'\n')
            keep = size - len(tail) + cut + 1 if cut >= 0 else 0
            f.seek(keep)
            with open(f'{path}.tail-{time.strftime("%Y%m%d-%H%M%S")}', 'wb') as g:
                g.write(f.read())
            f.truncate(keep)
        print(f'repaired {path}: removed a half-written last record ({size - keep} bytes)', flush=True)


def backup_journal(study):
    path = journal_path(study)
    if os.path.isfile(path):
        import shutil
        with _FileLock(path):
            shutil.copyfile(path, path + '.bak')


# --------------------------------------------------------------------------- #
# Interrupted trials (power cut, closed window, Ctrl+C)
# --------------------------------------------------------------------------- #
# The journal storage has no heartbeat, so a trial whose worker died stays
# RUNNING forever. Every worker registers itself in workers/<host>-<pid>.json;
# when a worker starts, RUNNING trials of workers that no longer exist are
# marked FAIL + 'interrupted' and their parameters are queued again. The
# interrupted attempts do not count towards --trials.

def _process_start_time(pid):
    ''' Start time (unix seconds) of a live process, or None if it is not running. '''
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:  # STILL_ACTIVE
                return None
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
                return 0.0
            ticks = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            return (ticks - 116444736000000000) / 1e7
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    return 0.0


def register_worker(study):
    import socket
    worker = f'{socket.gethostname()}-{os.getpid()}'
    folder = os.path.join(study_dir(study), 'workers')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, worker + '.json'), 'w', encoding='utf-8') as f:
        json.dump({'pid': os.getpid(), 'started': time.time()}, f)
    return worker


def worker_alive(study, worker):
    path = os.path.join(study_dir(study), 'workers', f'{worker}.json')
    try:
        with open(path, encoding='utf-8') as f:
            info = json.load(f)
    except (OSError, ValueError):
        return False
    started = _process_start_time(int(info['pid']))
    # a reused pid belongs to a process started after the worker registered
    return started is not None and started <= info['started'] + 5


def is_interrupted(t):
    return t.user_attrs.get('interrupted', False)


def count_finished(study):
    ''' Trials that count towards --trials: complete, pruned, and failed for
    a reason other than an interruption. '''
    from optuna.trial import TrialState
    n = 0
    for t in study.get_trials(deepcopy=False):
        if t.state in (TrialState.COMPLETE, TrialState.PRUNED):
            n += 1
        elif t.state == TrialState.FAIL and not is_interrupted(t):
            n += 1
    return n


def recover_interrupted(study_name, study, force=False):
    ''' Mark RUNNING trials of dead workers as interrupted and queue their
    parameters again (once). force=True treats every RUNNING trial as dead
    (--fail_stale; only when no worker of the study is running). '''
    from optuna.trial import TrialState
    with _FileLock(os.path.join(study_dir(study_name), 'recover')):
        stale = []
        for t in study.get_trials(deepcopy=False):
            if t.state != TrialState.RUNNING:
                continue
            worker = t.user_attrs.get('worker')
            if worker is None:      # created moments ago, or by a worker before this mechanism
                age = time.time() - t.datetime_start.timestamp() if t.datetime_start else 0
                dead = force or age > 600
            else:
                dead = force or not worker_alive(study_name, worker)
            if dead:
                study._storage.set_trial_user_attr(t._trial_id, 'interrupted', True)
                study._storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
                stale.append(t.number)
        requeued = []
        trials = study.get_trials(deepcopy=False)
        retried = {t.user_attrs.get('retry_of') for t in trials}
        for t in trials:
            if t.state == TrialState.FAIL and is_interrupted(t) and t.number not in retried and t.params:
                study.enqueue_trial(t.params, user_attrs={'retry_of': t.number})
                requeued.append(t.number)
    if stale:
        print(f'trial(s) {stale} were interrupted (their worker is gone): marked FAIL', flush=True)
    if requeued:
        print(f'queued the parameters of interrupted trial(s) {requeued} again', flush=True)


def load_study(study, cfg=None):
    import optuna
    cfg = cfg or load_config(study)
    sampler = optuna.samplers.TPESampler(n_startup_trials=cfg['startup_trials'],
                                         multivariate=True, constant_liar=True)
    if cfg['prune']:
        min_resource = max(1, round(cfg['min_resource_frac'] * cfg['evals']))
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=min_resource, reduction_factor=cfg['reduction_factor'])
    else:
        pruner = optuna.pruners.NopPruner()
    return optuna.create_study(study_name=study, storage=storage(study), direction='maximize',
                               sampler=sampler, pruner=pruner, load_if_exists=True)


# --------------------------------------------------------------------------- #
# Building a training run
# --------------------------------------------------------------------------- #

def build_args(cfg, params, log_dir, seed, save_model):
    ''' Training-script args = script defaults + FIXED + fixed space values +
    sampled `params` + study settings. Returns (module, args). '''
    module_name = tune_spaces.ALGORITHMS[cfg['algo']][0]
    module = importlib.import_module(module_name)
    args = module.build_parser().parse_args([])

    overrides = dict(cfg['fixed'])
    for name, spec in cfg['space'].items():
        value = params[name] if spec.get('tune', True) else spec['value']
        overrides.update(tune_spaces.to_args(name, spec, value))
    budget_arg = tune_spaces.ALGORITHMS[cfg['algo']][1]
    overrides.update({
        budget_arg: cfg['budget'],
        'evaluate_every': max(1, cfg['budget'] // cfg['evals']),
        'eval_deals': cfg['eval_deals'],
        'eval_panel': cfg['eval_panel'],
        'seed': seed,
        'log_dir': log_dir.rstrip('/\\') + '/',
        'save_model': save_model,
    })
    unknown = [k for k in overrides if not hasattr(args, k)]
    if unknown:
        raise ValueError(f'{module_name}.py has no argument(s) {unknown}')
    for key, value in overrides.items():
        setattr(args, key, value)
    os.makedirs(args.log_dir, exist_ok=True)
    return module, args


def suggest(trial, space):
    params = {}
    for name, spec in space.items():
        if not spec.get('tune', True):
            continue
        if spec['type'] == 'float':
            params[name] = trial.suggest_float(name, spec['low'], spec['high'], log=spec.get('log', False))
        elif spec['type'] == 'int':
            params[name] = trial.suggest_int(name, int(spec['low']), int(spec['high']),
                                             log=spec.get('log', False))
        else:
            params[name] = trial.suggest_categorical(name, list(spec['choices']))
    return params


def score(history, tail=3):
    return statistics.fmean(history[-tail:])


def _cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_run(a):
    import optuna
    from optuna.trial import TrialState
    warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)

    if a.space:
        with open(a.space, encoding='utf-8') as f:
            space = json.load(f)
    else:
        space = tune_spaces.default_space(a.algo)
    tune_spaces.validate_space(space)
    defaults = tune_spaces.DEFAULT_SETTINGS[a.algo]
    cfg, created = create_or_check_config(a.study, {
        'algo': a.algo,
        'budget': a.budget or defaults['budget'],
        'evals': a.evals or defaults['evals'],
        'eval_deals': a.eval_deals,
        'eval_panel': a.eval_panel,
        'seed': a.seed,
        'space': space,
        'fixed': tune_spaces.FIXED[a.algo],
        'prune': a.prune,
        'min_resource_frac': a.min_resource_frac,
        'reduction_factor': a.reduction_factor,
        'startup_trials': a.startup_trials,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    })
    study = load_study(a.study, cfg)

    # the first worker of a new study queues the space's starting configuration
    # as trial 0 (the other workers find the config already written)
    if created and a.enqueue:
        start = tune_spaces.start_params(cfg['space'])
        problems = check_start(cfg['space'], start) if start else ['a tuned parameter has no start value']
        if problems:
            print(f'starting configuration not queued: {"; ".join(problems)}')
        else:
            study.enqueue_trial(start, skip_if_exists=True)
            print(f'queued the starting configuration as the first trial: {json.dumps(start)}')

    backup_journal(a.study)
    worker = register_worker(a.study)
    recover_interrupted(a.study, study, force=a.fail_stale)

    def objective(trial):
        trial.set_user_attr('worker', worker)
        params = suggest(trial, cfg['space'])
        log_dir = os.path.join(study_dir(a.study), 'trials', f'trial_{trial.number}')
        trial.set_user_attr('log_dir', _rel(log_dir))
        print(f'\n=== trial {trial.number}: {json.dumps(params)}', flush=True)
        module, args = build_args(cfg, params, log_dir, cfg['seed'], save_model=False)
        history = []

        def on_eval(episode, results):
            history.append(results['min'])
            trial.report(results['min'], len(history) - 1)
            trial.set_user_attr('history', history)
            if trial.should_prune():
                print(f'=== trial {trial.number} pruned after {len(history)} evaluation(s) '
                      f'(episode {episode}, panel min {results["min"]:+.1f})', flush=True)
                raise optuna.TrialPruned()
            return False

        try:
            module.train(args, on_eval)
        except KeyboardInterrupt:
            # Ctrl+C / closing the worker: the trial is re-run when a worker starts again
            trial.set_user_attr('interrupted', True)
            raise
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            # typically a diverged network (NaN probabilities in np.random.choice);
            # after at least one report it is ended as PRUNED, so TPE still sees
            # its (bad) intermediate values instead of ignoring a FAILed trial
            if not history:
                raise
            trial.set_user_attr('diverged', repr(exc)[:300])
            print(f'=== trial {trial.number} stopped after {len(history)} evaluation(s): {exc!r}', flush=True)
            raise optuna.TrialPruned(f'diverged: {exc!r}')
        finally:
            _cleanup()
        if not history:
            raise RuntimeError('training finished without a panel evaluation')
        value = score(history)
        print(f'=== trial {trial.number} finished: score {value:+.1f}', flush=True)
        return value

    def stop_when_done(study, _trial):
        if count_finished(study) >= a.trials:
            study.stop()

    finished = count_finished(study)
    if finished >= a.trials:
        print(f'study already has {finished} finished trials (target {a.trials}); nothing to do')
        return
    print(f'worker {worker}: {finished}/{a.trials} trials finished', flush=True)
    try:
        study.optimize(objective, callbacks=[stop_when_done], catch=(Exception,), gc_after_trial=True)
    except KeyboardInterrupt:
        print('\nworker stopped; the interrupted trial will be re-run by the next worker started', flush=True)
        return
    print('worker done', flush=True)


def _trial_rows(study):
    rows = []
    for t in study.get_trials(deepcopy=False):
        steps = sorted(t.intermediate_values)
        rows.append({
            'number': t.number,
            'state': t.state.name,
            'value': t.value,
            'params': t.params,
            'evals': len(steps),
            'last_min': t.intermediate_values[steps[-1]] if steps else None,
            'seconds': (t.duration.total_seconds() if t.duration else
                        (time.time() - t.datetime_start.timestamp() if t.datetime_start else None)),
            'log_dir': t.user_attrs.get('log_dir'),
        })
    return rows


def cmd_status(a):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = load_config(a.study)
    study = load_study(a.study, cfg)
    rows = _trial_rows(study)
    finalists_file = os.path.join(study_dir(a.study), 'finalists', 'finalists.json')
    finalists = None
    if os.path.isfile(finalists_file):
        with open(finalists_file, encoding='utf-8') as f:
            finalists = json.load(f)
    if a.json:
        print(json.dumps({'config': cfg, 'trials': rows, 'finalists': finalists}, default=str))
        return
    counts = {}
    for r in rows:
        counts[r['state']] = counts.get(r['state'], 0) + 1
    print(f"study {a.study} ({cfg['algo']}, budget {cfg['budget']}, {cfg['evals']} evals/trial): {counts}")
    done = sorted((r for r in rows if r['value'] is not None), key=lambda r: -r['value'])
    for r in done[:a.top]:
        print(f"  trial {r['number']:3d}  {r['state']:8s} score {r['value']:+8.1f}  {json.dumps(r['params'])}")


def cmd_finalists(a):
    import optuna
    from optuna.trial import TrialState
    from eval_panel import Panel, parse_members
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    cfg = load_config(a.study)
    study = load_study(a.study, cfg)
    complete = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
    top = sorted(complete, key=lambda t: -t.value)[:a.top]
    if not top:
        raise SystemExit('no completed trials yet')

    out_dir = os.path.join(study_dir(a.study), 'finalists')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'finalists.json')
    previous = _read_finalists(out_file, a.final_deals)['final_deals']
    if previous != a.final_deals:
        raise SystemExit(f'{out_file} was produced with --final_deals {previous}; use the same value '
                         f'(or move the finalists folder away to start over)')
    worker = register_worker(a.study)

    seeds = [cfg['seed'] + 1000 + i for i in range(a.seeds)]
    for t in top:
        for seed in seeds:
            if (t.number, seed) in _finished_runs(out_file):
                print(f'trial {t.number} seed {seed}: already done, skipping')
                continue
            log_dir = os.path.join(out_dir, f'trial_{t.number}', f'seed_{seed}')
            if not _claim_run(a.study, log_dir, worker):
                print(f'trial {t.number} seed {seed}: being trained by another process, skipping')
                continue
            print(f'\n=== finalist trial {t.number} (score {t.value:+.1f}), seed {seed}: {json.dumps(t.params)}',
                  flush=True)
            module, args = build_args(cfg, t.params, log_dir, seed, save_model=True)
            history = []

            def track(episode, results):
                history.append(results['min'])
                return False   # never stop early

            try:
                agent = module.train(args, track)
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                # a diverged seed is a result too (the configuration is unstable): record it
                # and go on with the other runs instead of aborting the whole command
                run = {'trial': t.number, 'seed': seed, 'params': t.params, 'trial_score': t.value,
                       'failed': repr(exc)[:300], 'log_dir': _rel(log_dir)}
                _update_finalists(out_file, a.final_deals, lambda rec: rec['runs'].append(run))
                print(f'=== finalist trial {t.number} seed {seed} FAILED: {exc!r}', flush=True)
                _cleanup()
                continue
            # the trained agent reads state['obs'], so evaluate it in an env of its encoding
            results = Panel(parse_members(cfg['eval_panel']), num_deals=a.final_deals,
                            obs=getattr(args, 'obs', 'rlcard')).evaluate(agent)
            run = {'trial': t.number, 'seed': seed, 'params': t.params, 'trial_score': t.value,
                   'final_min': results['min'], 'final_mean': results['mean'],
                   'members': results['members'], 'training_score': score(history) if history else None,
                   'log_dir': _rel(log_dir)}
            _update_finalists(out_file, a.final_deals, lambda rec: rec['runs'].append(run))
            print(f"=== finalist trial {t.number} seed {seed}: final panel min {results['min']:+.1f} "
                  f"(mean {results['mean']:+.1f}) over {a.final_deals} deals", flush=True)
            del agent
            _cleanup()

    record = _read_finalists(out_file, a.final_deals)
    missing = [(t.number, s) for t in top for s in seeds
               if (t.number, s) not in {(r['trial'], r['seed']) for r in record['runs']}]
    if missing:
        print(f'\n{len(missing)} finalist run(s) still running in other processes or not done: {missing}; '
              f'the last process to finish picks the winner')
        return
    print('\nFinalists (final panel min over seeds; configurations with failed seeds rank last):')
    summary = []
    for t in top:
        runs = [r for r in record['runs'] if r['trial'] == t.number and r['seed'] in seeds]
        mins = [r['final_min'] for r in runs if 'failed' not in r]
        failed = len(runs) - len(mins)
        if not mins:
            print(f'  trial {t.number:3d}: every seed failed')
            continue
        sd = statistics.stdev(mins) if len(mins) > 1 else float('nan')
        summary.append((failed, statistics.fmean(mins), sd, len(mins), t))
    summary.sort(key=lambda x: (x[0], -x[1]))
    for failed, mean, sd, n, t in summary:
        print(f'  trial {t.number:3d}: {mean:+8.1f} +/- {sd:5.1f} (sd over {n} seeds'
              + (f', {failed} failed' if failed else '') + f')  {json.dumps(t.params)}')
    if summary:
        best = summary[0][4]
        good = [r['seed'] for r in record['runs'] if r['trial'] == best.number and r['seed'] in seeds
                and 'failed' not in r]
        winner = {'trial': best.number, 'params': best.params, 'final_min_mean': summary[0][1],
                  'seeds': sorted(good), 'final_deals': a.final_deals}
        _update_finalists(out_file, a.final_deals, lambda rec: rec.update(winner=winner))
        print(f'\nwinner: trial {best.number}')


def _rel(path):
    ''' Repo-relative path with forward slashes (works on Windows and Linux). '''
    return os.path.relpath(path, REPO_ROOT).replace('\\', '/')


def _read_finalists(path, final_deals):
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {'final_deals': final_deals, 'runs': []}


def _finished_runs(path):
    return {(r['trial'], r['seed']) for r in _read_finalists(path, None)['runs']}


def _update_finalists(path, final_deals, change):
    ''' Read-modify-write finalists.json under a lock (several finalists
    processes may run at once); written atomically via a temporary file. '''
    with _FileLock(path):
        record = _read_finalists(path, final_deals)
        change(record)
        with open(path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2)
        os.replace(path + '.tmp', path)


def _claim_run(study, log_dir, worker):
    ''' Claim a finalist run for this process: False if a live process has
    it. A run left half-done by a dead process is restarted from scratch. '''
    import shutil
    claim = os.path.join(log_dir, 'claimed_by.txt')
    with _FileLock(os.path.join(study_dir(study), 'finalists', 'claims')):
        if os.path.isfile(claim):
            with open(claim, encoding='utf-8') as f:
                owner = f.read().strip()
            if owner != worker and worker_alive(study, owner):
                return False
            print(f'restarting {os.path.relpath(log_dir, REPO_ROOT)} (left unfinished by {owner})', flush=True)
            shutil.rmtree(log_dir, ignore_errors=True)
        os.makedirs(log_dir, exist_ok=True)
        with open(claim, 'w', encoding='utf-8') as f:
            f.write(worker)
    return True


def cmd_report(a):
    import optuna
    from optuna.trial import TrialState
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import optuna.visualization.matplotlib as vis
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
    cfg = load_config(a.study)
    study = load_study(a.study, cfg)
    plot_dir = os.path.join(study_dir(a.study), 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    complete = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
    print(f'{len(study.trials)} trials, {len(complete)} complete')
    if complete:
        best = study.best_trial
        print(f'best trial {best.number}: score {best.value:+.1f}  {json.dumps(best.params)}')

    plots = {
        'optimization_history': lambda: vis.plot_optimization_history(study),
        'intermediate_values': lambda: vis.plot_intermediate_values(study),
        'parallel_coordinate': lambda: vis.plot_parallel_coordinate(study),
        'slice': lambda: vis.plot_slice(study),
        'param_importances': lambda: vis.plot_param_importances(study),
    }
    for name, make in plots.items():
        try:
            make()
            fig = plt.gcf()
            fig.set_size_inches(11, 6)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f'{name}.png'), dpi=110)
            print(f'  wrote plots/{name}.png')
        except Exception as exc:   # e.g. too few complete trials for importances
            print(f'  skipped {name}: {exc}')
        finally:
            plt.close('all')

    if len(complete) >= 3:
        try:
            importances = optuna.importance.get_param_importances(study)
            print('parameter importances (fANOVA):')
            for k, v in importances.items():
                print(f'  {k:28s} {v:.3f}')
        except Exception as exc:
            print(f'  importances unavailable: {exc}')


def main():
    p = argparse.ArgumentParser('Optuna hyper-parameter tuning for the limit hold\'em agents')
    sub = p.add_subparsers(dest='command', required=True)

    r = sub.add_parser('run', help='run trials (start several processes to work in parallel)')
    r.add_argument('--algo', required=True, choices=sorted(tune_spaces.ALGORITHMS))
    r.add_argument('--study', required=True)
    r.add_argument('--trials', type=int, default=40,
                   help='total trials of the study (across all workers), incl. pruned/failed')
    r.add_argument('--budget', type=int, default=None,
                   help='episodes (Deep CFR: iterations) per trial; default per algorithm')
    r.add_argument('--evals', type=int, default=None, help='panel evaluations per trial')
    r.add_argument('--eval_deals', type=int, default=500, help='duplicate deals per panel member')
    r.add_argument('--eval_panel', type=str, default='random,station,maniac,nit,tag,lag')
    r.add_argument('--seed', type=int, default=42, help='training seed of every trial')
    r.add_argument('--space', type=str, default=None, help='JSON search space (default: tune_spaces.py)')
    r.add_argument('--prune', type=int, default=1, help='1 = successive-halving pruning, 0 = off')
    r.add_argument('--min_resource_frac', type=float, default=0.25,
                   help='fraction of a trial\'s evaluations before it can be pruned')
    r.add_argument('--reduction_factor', type=int, default=3)
    r.add_argument('--startup_trials', type=int, default=12, help='random trials before TPE kicks in')
    r.add_argument('--enqueue', type=int, default=1,
                   help="1 = a new study's first trial uses the space's 'start' values "
                        '(tune_spaces.py), 0 = all startup trials random')
    r.add_argument('--threads', type=int, default=0, help='torch threads (0 = torch default)')
    r.add_argument('--fail_stale', action='store_true',
                   help='treat every RUNNING trial as interrupted (only when no worker of the study runs; '
                        'normally not needed: trials of dead workers are detected automatically)')

    s = sub.add_parser('status', help='summary of a study')
    s.add_argument('--study', required=True)
    s.add_argument('--json', action='store_true')
    s.add_argument('--top', type=int, default=10)

    f = sub.add_parser('finalists', help='re-run the best trials with new seeds and a long evaluation')
    f.add_argument('--study', required=True)
    f.add_argument('--top', type=int, default=3)
    f.add_argument('--seeds', type=int, default=3)
    f.add_argument('--final_deals', type=int, default=5000)
    f.add_argument('--threads', type=int, default=0)

    g = sub.add_parser('report', help='write plots and print the best trial / importances')
    g.add_argument('--study', required=True)

    a = p.parse_args()
    {'run': cmd_run, 'status': cmd_status, 'finalists': cmd_finalists, 'report': cmd_report}[a.command](a)


if __name__ == '__main__':
    main()
