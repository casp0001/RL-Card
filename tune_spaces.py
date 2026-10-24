''' Hyper-parameter search spaces for tune.py (plain Python, no third-party
imports, so tune_gui.py can use it from any interpreter).

A space maps a parameter name to a spec dict:
    {'type': 'float' | 'int', 'low': ..., 'high': ..., 'log': bool}
    {'type': 'categorical', 'choices': [...]}
plus optional keys
    'targets': [arg, ...]   script arguments that receive the value
                            (default: the parameter name itself)
    'list': True            a categorical whose choices are strings of ints,
                            e.g. '256 256' -> [256, 256] (layer sizes)
    'tune': False           do not search, use 'value' instead
    'value': ...            the fixed value when 'tune' is False
    'start': ...            the starting value: tune.py --enqueue 1 runs it as a
                            study's first trial (a known-sensible configuration,
                            so TPE has a reasonable point from the start)
    'help': '...'

FIXED holds script arguments that are always overridden for tuning trials.

Why some parameters are fixed:
    PPO gae_lambda      with gamma = 1 and at most ~4 decisions per hand per
                        player, lambda hardly changes the advantages
    Deep CFR num_traversals / num_step
                        the budget counts iterations, so more traversals or SGD
                        steps per iteration are simply more compute; tuning them
                        would drift to the upper bounds instead of finding a
                        better configuration at equal cost
'''

import copy


ALGORITHMS = {
    # name: (training module, budget argument, unit of the budget)
    'dqn': ('limit_dqn_selfplay_custom', 'num_episodes', 'episodes'),
    'ppo': ('limit_ppo_selfplay', 'num_episodes', 'episodes'),
    'nfsp': ('limit_nfsp_selfplay_custom_hyperparams', 'num_episodes', 'episodes'),
    'deepcfr': ('limit_deepcfr_train_custom', 'num_episodes', 'iterations'),
}

LAYERS = ['128 128', '256 256', '256 256 256', '512 512']

SPACES = {
    # DQN ranges widened after the PPO studies (50 trials per study): in PPO the
    # exploration parameter (entropy) mattered most and its best region moved
    # with the training protocol, the opponent-snapshot interval was the next
    # most important (best 3k-15k, mid-range) and the best learning rates hit
    # the top of their range -- so DQN tunes its exploration (epsilon_end too,
    # previously fixed) and gets wider learning-rate / interval ranges.
    'dqn': {
        'lr': {'type': 'float', 'low': 1e-5, 'high': 3e-3, 'log': True, 'start': 1e-4},
        'batch_size': {'type': 'categorical', 'choices': [32, 64, 128, 256, 512], 'start': 128},
        'mlp_layers': {'type': 'categorical', 'choices': ['128 128', '256 256', '512 512', '256 256 256',
                                                          '512 512 512'], 'list': True, 'start': '256 256'},
        'epsilon_decay_steps': {'type': 'int', 'low': 50_000, 'high': 3_000_000, 'log': True,
                                'start': 500_000,
                                'help': "counted in the learning agent's decisions (a few per episode)"},
        'epsilon_end': {'type': 'float', 'low': 0.01, 'high': 0.25, 'log': True, 'start': 0.05,
                        'help': 'exploration left after the decay (the DQN counterpart of PPO\'s entropy bonus)'},
        'replay_memory_size': {'type': 'int', 'low': 20_000, 'high': 1_000_000, 'log': True, 'start': 200_000,
                               'help': '~0.6 KB per transition'},
        'opponent_update_every': {'type': 'int', 'low': 1_000, 'high': 100_000, 'log': True, 'start': 10_000},
        'update_target_every': {'type': 'int', 'low': 250, 'high': 10_000, 'log': True, 'start': 1_000,
                                'help': 'training steps between target-network copies'},
        'train_every': {'type': 'categorical', 'choices': [1, 2, 4], 'tune': False, 'value': 1},
        'activation': {'type': 'categorical', 'choices': ['relu', 'tanh'], 'tune': False, 'value': 'relu'},
    },
    'ppo': {
        'lr': {'type': 'float', 'low': 5e-5, 'high': 1e-3, 'log': True, 'start': 3e-4},
        'update_every': {'type': 'categorical', 'choices': [2048, 4096, 8192, 16384], 'start': 4096},
        'num_epochs': {'type': 'int', 'low': 2, 'high': 8, 'log': False, 'start': 4},
        'batch_size': {'type': 'categorical', 'choices': [128, 256, 512, 1024], 'start': 256},
        'clip_range': {'type': 'float', 'low': 0.1, 'high': 0.3, 'log': False, 'start': 0.2},
        'entropy_coef': {'type': 'float', 'low': 1e-3, 'high': 3e-2, 'log': True, 'start': 0.01},
        'gae_lambda': {'type': 'float', 'low': 0.9, 'high': 0.99, 'log': False, 'tune': False, 'value': 0.95},
        'mlp_layers': {'type': 'categorical', 'choices': LAYERS, 'list': True, 'tune': False,
                       'value': '256 256'},
        'opponent_update_every': {'type': 'int', 'low': 2_000, 'high': 50_000, 'log': True, 'start': 10_000,
                                  'help': 'self-play stability is the usual PPO failure in poker'},
    },
    # NFSP ranges widened after the PPO / DQN studies: exploration and the
    # opponent-snapshot interval mattered most in PPO, so the epsilon range is
    # wider and the snapshot interval is tuned; the learning-rate ranges are
    # wider on both sides. Network size and train_every set the compute per
    # episode (DQN showed a 5-8x spread), so experiments/nfsp_pipeline.py
    # measures them in a pilot and drops choices that are too expensive.
    # Buffer ranges are capped so 4 workers fit in RAM (~1.5 GB each when full).
    'nfsp': {
        # starts follow Heinrich & Silver (2016): RL 0.1, SL 0.005, eta 0.1, epsilon 0.06
        'rl_learning_rate': {'type': 'float', 'low': 2e-3, 'high': 0.3, 'log': True, 'start': 0.05,
                             'help': 'range for SGD; use ~1e-5..1e-3 with optimizer=adam'},
        'sl_learning_rate': {'type': 'float', 'low': 5e-4, 'high': 5e-2, 'log': True, 'start': 0.005,
                             'help': 'range for SGD; use ~1e-5..1e-3 with optimizer=adam'},
        'anticipatory_param': {'type': 'float', 'low': 0.03, 'high': 0.5, 'log': True, 'start': 0.1},
        'reservoir_buffer_capacity': {'type': 'int', 'low': 200_000, 'high': 1_500_000, 'log': True,
                                      'start': 1_000_000, 'help': '~0.3 KB per entry'},
        'q_replay_memory_size': {'type': 'int', 'low': 100_000, 'high': 600_000, 'log': True, 'start': 200_000,
                                 'help': '~0.6 KB per entry'},
        'q_epsilon_start': {'type': 'float', 'low': 0.02, 'high': 0.3, 'log': True, 'start': 0.06},
        'layers': {'type': 'categorical', 'choices': ['128 128', '256 256', '512 512', '256 256 256'],
                   'list': True, 'targets': ['hidden_layers_sizes', 'q_mlp_layers'], 'start': '256 256'},
        'train_every': {'type': 'categorical', 'choices': [32, 64, 128, 256],
                        'targets': ['train_every', 'q_train_every'], 'start': 128},
        'optimizer': {'type': 'categorical', 'choices': ['sgd', 'adam'], 'tune': False, 'value': 'sgd'},
        'q_activation': {'type': 'categorical', 'choices': ['relu', 'tanh'], 'tune': False, 'value': 'relu'},
        'opponent_update_every': {'type': 'int', 'low': 1_000, 'high': 100_000, 'log': True, 'start': 10_000},
    },
    'deepcfr': {
        'lr': {'type': 'float', 'low': 1e-4, 'high': 3e-3, 'log': True, 'start': 5e-4},
        # ~60 s per iteration at 1000 traversals + 1000 SGD steps (batch 4096, 4 threads).
        # Fixed (see the module docstring): they set the compute per iteration.
        'num_traversals': {'type': 'int', 'low': 100, 'high': 1500, 'log': True, 'tune': False, 'value': 1000},
        'num_step': {'type': 'int', 'low': 200, 'high': 2000, 'log': True, 'tune': False, 'value': 1000},
        'batch_size': {'type': 'categorical', 'choices': [1024, 4096, 8192],
                       'targets': ['batch_size_advantage', 'batch_size_strategy'], 'start': 4096},
        'layers': {'type': 'categorical', 'choices': ['128 128 128', '256 256 256'], 'list': True,
                   'targets': ['policy_layers', 'advantage_layers'], 'start': '256 256 256'},
        'memory_capacity': {'type': 'int', 'low': 500_000, 'high': 5_000_000, 'log': True,
                            'tune': False, 'value': 2_000_000,
                            'help': 'per buffer (3 buffers, ~0.3 KB per entry)'},
    },
}

FIXED = {
    # every algorithm trains on the rich observation (obs_encoding.py)
    # gamma = 1 for every algorithm: a hand pays out only at its end, so
    # discounting would just shrink values by the number of decisions left
    # (DQN's epsilon_end is tuned in its space; it used to be fixed at 0.05 here)
    'dqn': {'verbose': False, 'obs': 'rich', 'replay_memory_init_size': 5_000, 'discount_factor': 1.0},
    'ppo': {'verbose': False, 'obs': 'rich', 'discount_factor': 1.0},
    'nfsp': {'verbose': False, 'obs': 'rich', 'q_discount_factor': 1.0, 'eval_best_response': False,
             'q_replay_memory_init_size': 5_000, 'min_buffer_size_to_learn': 5_000},
    'deepcfr': {'verbose': False, 'obs': 'rich'},
}

# Default study settings per algorithm: budget per trial (in the algorithm's
# unit) and how many panel evaluations a trial reports. 10 evaluations of
# 6 x 500 duplicate deals cost ~3-4 minutes per trial (~2 % of its time) and
# still give the pruner rungs at ~30 % and ~70 % of the budget.
DEFAULT_SETTINGS = {
    'dqn': {'budget': 1_000_000, 'evals': 10},
    'ppo': {'budget': 1_000_000, 'evals': 10},
    'nfsp': {'budget': 2_000_000, 'evals': 10},
    'deepcfr': {'budget': 200, 'evals': 10},
}


def default_space(algo):
    return copy.deepcopy(SPACES[algo])


def validate_space(space):
    ''' Raise ValueError if a space is malformed. '''
    for name, spec in space.items():
        kind = spec.get('type')
        if kind in ('float', 'int'):
            low, high = spec.get('low'), spec.get('high')
            if low is None or high is None or not low < high:
                raise ValueError(f'{name}: need low < high, got {low}, {high}')
            if spec.get('log') and low <= 0:
                raise ValueError(f'{name}: log scale needs low > 0')
        elif kind == 'categorical':
            if not spec.get('choices'):
                raise ValueError(f'{name}: categorical needs choices')
        else:
            raise ValueError(f'{name}: unknown type {kind!r}')
        if spec.get('tune', True) is False and 'value' not in spec:
            raise ValueError(f'{name}: fixed parameter needs a value')


def to_args(name, spec, value):
    ''' Convert a sampled / fixed value into {script argument: value}. '''
    if spec.get('list'):
        value = [int(v) for v in str(value).split()]
    elif spec.get('type') == 'int':
        value = int(value)
    elif spec.get('type') == 'float':
        value = float(value)
    return {target: value for target in spec.get('targets', [name])}


def start_params(space):
    ''' The starting configuration of a space ({name: value} of the tuned
    parameters), or None if a tuned parameter has no 'start'. '''
    params = {}
    for name, spec in space.items():
        if spec.get('tune', True) is False:
            continue
        if 'start' not in spec:
            return None
        params[name] = spec['start']
    return params


# Bytes per stored sample of the array buffers (array_buffers.py) with the
# 153-value rich observation stored as float16 (the 72-value one is smaller).
OBS_BYTES = 153 * 2
DQN_TRANSITION_BYTES = 2 * OBS_BYTES + 8 + 4 + 1 + 4         # state, next_state, action, reward, done, legal
NFSP_RESERVOIR_BYTES = OBS_BYTES + 4 * 2                     # info_state, action_probs (float16)
DEEPCFR_BYTES = (OBS_BYTES + 4 + 16) + 2 * (OBS_BYTES + 4 + 4 + 2)   # strategy + 2 advantage buffers
PROCESS_OVERHEAD_GB = 0.6                                    # python + torch + rlcard


def estimate_memory_gb(algo, space):
    ''' Rough upper bound of one trial's memory in GB: the replay / reservoir
    buffers when full (they grow as they fill) plus the process itself. '''
    def upper(name, default):
        spec = space.get(name)
        if spec is None:
            return default
        if spec.get('tune', True) is False:
            return float(spec['value'])
        return float(spec.get('high', default))
    # opponent snapshots are frozen_copy()s without buffers, so no factor 2
    if algo == 'dqn':
        buffers = upper('replay_memory_size', 40_000) * DQN_TRANSITION_BYTES
    elif algo == 'nfsp':
        buffers = (upper('q_replay_memory_size', 200_000) * DQN_TRANSITION_BYTES
                   + upper('reservoir_buffer_capacity', 1_500_000) * NFSP_RESERVOIR_BYTES)
    elif algo == 'deepcfr':
        buffers = upper('memory_capacity', 2_000_000) * DEEPCFR_BYTES
    else:
        buffers = 0.1e9   # ppo: the on-policy buffer is cleared every update
    return buffers / 1e9 + PROCESS_OVERHEAD_GB
