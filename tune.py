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
    existing one (all workers of a study must use the same settings). '''
    os.makedirs(study_dir(study), exist_ok=True)
    path = config_path(study)
    try:
        with open(path, 'x', encoding='utf-8') as f:   # 'x': fails if it exists
            json.dump(cfg, f, indent=2)
        return cfg
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
    return existing


def storage(study):
    import optuna
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    path = os.path.join(study_dir(study), 'journal.log')
    # The default symlink lock needs extra privileges on Windows.
    return optuna.storages.JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


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
    cfg = create_or_check_config(a.study, {
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

    if a.fail_stale:
        stale = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.RUNNING]
        for t in stale:
            study._storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
        if stale:
            print(f'marked {len(stale)} interrupted trial(s) as FAIL: {[t.number for t in stale]}')

    def objective(trial):
        params = suggest(trial, cfg['space'])
        log_dir = os.path.join(study_dir(a.study), 'trials', f'trial_{trial.number}')
        trial.set_user_attr('log_dir', os.path.relpath(log_dir, REPO_ROOT))
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
        finally:
            _cleanup()
        if not history:
            raise RuntimeError('training finished without a panel evaluation')
        value = score(history)
        print(f'=== trial {trial.number} finished: score {value:+.1f}', flush=True)
        return value

    states = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    finished = len([t for t in study.get_trials(deepcopy=False) if t.state in states])
    if finished >= a.trials:
        print(f'study already has {finished} finished trials (target {a.trials}); nothing to do')
        return
    study.optimize(objective, callbacks=[optuna.study.MaxTrialsCallback(a.trials, states=states)],
                   catch=(Exception,), gc_after_trial=True)
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
    record = {'final_deals': a.final_deals, 'runs': []}
    if os.path.isfile(out_file):
        with open(out_file, encoding='utf-8') as f:
            record = json.load(f)
    done = {(r['trial'], r['seed']) for r in record['runs']}

    seeds = [cfg['seed'] + 1000 + i for i in range(a.seeds)]
    panel = Panel(parse_members(cfg['eval_panel']), num_deals=a.final_deals)
    for t in top:
        for seed in seeds:
            if (t.number, seed) in done:
                print(f'trial {t.number} seed {seed}: already done, skipping')
                continue
            print(f'\n=== finalist trial {t.number} (score {t.value:+.1f}), seed {seed}: {json.dumps(t.params)}',
                  flush=True)
            log_dir = os.path.join(out_dir, f'trial_{t.number}', f'seed_{seed}')
            module, args = build_args(cfg, t.params, log_dir, seed, save_model=True)
            history = []

            def track(episode, results):
                history.append(results['min'])
                return False   # never stop early

            agent = module.train(args, track)
            results = panel.evaluate(agent)
            run = {'trial': t.number, 'seed': seed, 'params': t.params, 'trial_score': t.value,
                   'final_min': results['min'], 'final_mean': results['mean'],
                   'members': results['members'], 'training_score': score(history) if history else None,
                   'log_dir': os.path.relpath(log_dir, REPO_ROOT)}
            record['runs'].append(run)
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2)
            print(f"=== finalist trial {t.number} seed {seed}: final panel min {results['min']:+.1f} "
                  f"(mean {results['mean']:+.1f}) over {a.final_deals} deals", flush=True)
            del agent
            _cleanup()

    print('\nFinalists (final panel min over seeds):')
    summary = []
    for t in top:
        mins = [r['final_min'] for r in record['runs'] if r['trial'] == t.number]
        if not mins:
            continue
        sd = statistics.stdev(mins) if len(mins) > 1 else float('nan')
        summary.append((statistics.fmean(mins), sd, len(mins), t))
    summary.sort(key=lambda x: -x[0])
    for mean, sd, n, t in summary:
        print(f'  trial {t.number:3d}: {mean:+8.1f} +/- {sd:5.1f} (sd over {n} seeds)  {json.dumps(t.params)}')
    if summary:
        best = summary[0][3]
        record['winner'] = {'trial': best.number, 'params': best.params, 'final_min_mean': summary[0][0]}
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2)
        print(f'\nwinner: trial {best.number}')


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
    r.add_argument('--threads', type=int, default=0, help='torch threads (0 = torch default)')
    r.add_argument('--fail_stale', action='store_true',
                   help='mark RUNNING trials left by killed workers as FAIL (only when no worker runs)')

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
