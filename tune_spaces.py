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
    'help': '...'

FIXED holds script arguments that are always overridden for tuning trials.
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
    'dqn': {
        'lr': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True},
        'batch_size': {'type': 'categorical', 'choices': [64, 128, 256]},
        'mlp_layers': {'type': 'categorical', 'choices': LAYERS, 'list': True},
        'epsilon_decay_steps': {'type': 'int', 'low': 100_000, 'high': 2_000_000, 'log': True},
        'replay_memory_size': {'type': 'int', 'low': 50_000, 'high': 500_000, 'log': True},
        'opponent_update_every': {'type': 'int', 'low': 2_000, 'high': 50_000, 'log': True},
        'train_every': {'type': 'categorical', 'choices': [1, 2, 4], 'tune': False, 'value': 1},
        'activation': {'type': 'categorical', 'choices': ['relu', 'tanh'], 'tune': False, 'value': 'relu'},
    },
    'ppo': {
        'lr': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True},
        'update_every': {'type': 'categorical', 'choices': [2048, 4096, 8192, 16384]},
        'num_epochs': {'type': 'int', 'low': 2, 'high': 8, 'log': False},
        'batch_size': {'type': 'categorical', 'choices': [128, 256, 512, 1024]},
        'clip_range': {'type': 'float', 'low': 0.1, 'high': 0.3, 'log': False},
        'entropy_coef': {'type': 'float', 'low': 1e-4, 'high': 3e-2, 'log': True},
        'gae_lambda': {'type': 'float', 'low': 0.9, 'high': 0.99, 'log': False},
        'mlp_layers': {'type': 'categorical', 'choices': LAYERS, 'list': True, 'tune': False,
                       'value': '256 256'},
        'opponent_update_every': {'type': 'int', 'low': 2_000, 'high': 50_000, 'log': True,
                                  'tune': False, 'value': 10_000},
    },
    'nfsp': {
        'rl_learning_rate': {'type': 'float', 'low': 1e-3, 'high': 0.2, 'log': True,
                             'help': 'range for SGD; use ~1e-5..1e-3 with optimizer=adam'},
        'sl_learning_rate': {'type': 'float', 'low': 1e-4, 'high': 2e-2, 'log': True,
                             'help': 'range for SGD; use ~1e-5..1e-3 with optimizer=adam'},
        'anticipatory_param': {'type': 'float', 'low': 0.05, 'high': 0.3, 'log': False},
        'reservoir_buffer_capacity': {'type': 'int', 'low': 200_000, 'high': 2_000_000, 'log': True,
                                      'help': '~0.9 KB per entry until the memory fixes land'},
        'q_replay_memory_size': {'type': 'int', 'low': 100_000, 'high': 600_000, 'log': True},
        'q_epsilon_start': {'type': 'float', 'low': 0.02, 'high': 0.12, 'log': False},
        'layers': {'type': 'categorical', 'choices': ['256 256', '512 512', '256 256 256'],
                   'list': True, 'targets': ['hidden_layers_sizes', 'q_mlp_layers']},
        'train_every': {'type': 'categorical', 'choices': [64, 128, 256],
                        'targets': ['train_every', 'q_train_every']},
        'optimizer': {'type': 'categorical', 'choices': ['sgd', 'adam'], 'tune': False, 'value': 'sgd'},
        'q_activation': {'type': 'categorical', 'choices': ['relu', 'tanh'], 'tune': False, 'value': 'relu'},
        'opponent_update_every': {'type': 'int', 'low': 1_000, 'high': 50_000, 'log': True,
                                  'tune': False, 'value': 10_000},
    },
    'deepcfr': {
        'lr': {'type': 'float', 'low': 1e-4, 'high': 3e-3, 'log': True},
        'num_traversals': {'type': 'int', 'low': 200, 'high': 3000, 'log': True},
        'num_step': {'type': 'int', 'low': 200, 'high': 4000, 'log': True},
        'batch_size': {'type': 'categorical', 'choices': [1024, 4096, 8192],
                       'targets': ['batch_size_advantage', 'batch_size_strategy']},
        'layers': {'type': 'categorical', 'choices': ['128 128 128', '256 256 256'], 'list': True,
                   'targets': ['policy_layers', 'advantage_layers']},
        'memory_capacity': {'type': 'int', 'low': 500_000, 'high': 5_000_000, 'log': True,
                            'tune': False, 'value': 2_000_000,
                            'help': 'per buffer (3 buffers, ~0.8 KB per entry until the memory fixes land)'},
    },
}

FIXED = {
    'dqn': {'verbose': False, 'replay_memory_init_size': 5_000},
    'ppo': {'verbose': False, 'discount_factor': 1.0},
    'nfsp': {'verbose': False, 'q_discount_factor': 1.0, 'eval_best_response': False,
             'q_replay_memory_init_size': 5_000, 'min_buffer_size_to_learn': 5_000},
    'deepcfr': {'verbose': False},
}

# Default study settings per algorithm: budget per trial (in the algorithm's
# unit) and how many panel evaluations a trial reports.
DEFAULT_SETTINGS = {
    'dqn': {'budget': 1_000_000, 'evals': 20},
    'ppo': {'budget': 1_000_000, 'evals': 20},
    'nfsp': {'budget': 2_000_000, 'evals': 20},
    'deepcfr': {'budget': 200, 'evals': 20},
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


def estimate_memory_gb(algo, space):
    ''' Rough upper bound of one trial's replay / reservoir memory in GB
    (current list-based buffers; measured bytes per stored sample). '''
    def upper(name, default):
        spec = space.get(name)
        if spec is None:
            return default
        if spec.get('tune', True) is False:
            return float(spec['value'])
        return float(spec.get('high', default))
    if algo == 'dqn':
        return 2 * upper('replay_memory_size', 40_000) * 1576 / 1e9          # + opponent snapshot copy
    if algo == 'nfsp':
        return 2 * (upper('q_replay_memory_size', 200_000) * 1576
                    + upper('reservoir_buffer_capacity', 1_500_000) * 913) / 1e9
    if algo == 'deepcfr':
        return upper('memory_capacity', 2_000_000) * (2 * 785 + 921) / 1e9
    return 0.1   # ppo: on-policy buffer is cleared every update
