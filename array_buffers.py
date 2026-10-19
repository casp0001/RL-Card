''' Compact replay / reservoir buffers backed by numpy arrays.

The original buffers stored every sample as a namedtuple of separate float64
numpy arrays (~0.8-1.6 KB per sample, mostly Python object overhead). These
store each field in one preallocated array (observations as float16: exact
for RLCard's 0/1 encoding, and fine for small numeric features), about 10x
smaller, and sample a batch with one fancy-indexing operation.

Arrays grow on demand (doubling) up to the capacity, so a large capacity does
not reserve memory up front.

    Reservoir   uniform sample of everything ever added (NFSP SL memory, Deep CFR)
    Ring        the most recent `capacity` items (DQN replay memory)
    NullBuffer  accepts and discards everything (for frozen opponent snapshots)
'''

import random

import numpy as np


class _ArrayStore(object):

    def __init__(self, capacity, fields=None, initial=4096):
        ''' fields: {name: (shape tuple, dtype)}; may be given later (lazily
        on the first add) via _ensure_fields. '''
        self.capacity = int(capacity)
        self.size = 0
        self._initial = initial
        self.fields = None
        self.data = None
        if fields is not None:
            self._make(fields)

    def _make(self, fields):
        self.fields = {k: (tuple(shape), np.dtype(dtype)) for k, (shape, dtype) in fields.items()}
        n = min(self.capacity, self._initial)
        self.data = {k: np.zeros((n,) + shape, dtype) for k, (shape, dtype) in self.fields.items()}

    def _reserve(self, n):
        current = len(next(iter(self.data.values())))
        if n <= current:
            return
        new = min(self.capacity, max(n, 2 * current))
        for k, arr in self.data.items():
            grown = np.zeros((new,) + arr.shape[1:], arr.dtype)
            grown[:current] = arr
            self.data[k] = grown

    def _write(self, i, row):
        for k, v in row.items():
            self.data[k][i] = v

    def __len__(self):
        return self.size

    def nbytes(self):
        return sum(a[:self.size].nbytes for a in self.data.values()) if self.data else 0

    def sample(self, n):
        ''' n distinct items, uniformly at random: {field: array}. Uses
        Python's `random` (seeded by rlcard.utils.set_seed). '''
        if n > self.size:
            raise ValueError(f'{n} elements could not be sampled from size {self.size}')
        idx = np.fromiter(random.sample(range(self.size), n), dtype=np.int64, count=n)
        return {k: arr[idx] for k, arr in self.data.items()}

    def arrays(self):
        return {k: arr[:self.size] for k, arr in self.data.items()} if self.data else {}

    def clear(self):
        self.size = 0

    def _state(self):
        return {'capacity': self.capacity, 'size': self.size,
                'fields': {k: (list(s), d.str) for k, (s, d) in (self.fields or {}).items()},
                'arrays': {k: a.copy() for k, a in self.arrays().items()}}

    def _load(self, state):
        if state['fields']:
            self._make({k: (tuple(s), d) for k, (s, d) in state['fields'].items()})
            self._reserve(state['size'])
            for k, a in state['arrays'].items():
                self.data[k][:state['size']] = a
        self.size = state['size']


class Reservoir(_ArrayStore):
    ''' Reservoir sampling: once full, the n-th item replaces a uniformly
    random slot with probability capacity / n. '''

    def __init__(self, capacity, fields=None, initial=4096):
        super().__init__(capacity, fields, initial)
        self.add_calls = 0

    def add(self, row):
        if self.data is None:
            raise RuntimeError('fields not set')
        if self.size < self.capacity:
            self._reserve(self.size + 1)
            self._write(self.size, row)
            self.size += 1
        else:
            j = random.randint(0, self.add_calls)
            if j < self.capacity:
                self._write(j, row)
        self.add_calls += 1

    def clear(self):
        self.size = 0
        self.add_calls = 0

    def checkpoint_attributes(self, include_data=True):
        state = self._state() if include_data else {**self._state(), 'size': 0, 'arrays': {}}
        state['add_calls'] = self.add_calls if include_data else 0
        state['kind'] = 'reservoir'
        return state

    @classmethod
    def from_state(cls, state):
        buf = cls(state['capacity'])
        buf._load(state)
        buf.add_calls = state.get('add_calls', buf.size)
        return buf


class Ring(_ArrayStore):
    ''' FIFO ring buffer: the newest item overwrites the oldest. '''

    def __init__(self, capacity, fields=None, initial=4096):
        super().__init__(capacity, fields, initial)
        self.next = 0

    def add(self, row):
        if self.size < self.capacity:
            self._reserve(self.size + 1)
            self._write(self.size, row)
            self.size += 1
        else:
            self._write(self.next, row)
            self.next = (self.next + 1) % self.capacity

    def clear(self):
        self.size = 0
        self.next = 0


class NullBuffer(object):
    ''' Stand-in for a replay / reservoir buffer that stores nothing. Used for
    frozen opponent snapshots, which act but never train. '''

    memory = ()      # DQN code checks len(memory.memory)

    def add(self, *args, **kwargs):
        pass

    save = add

    def __len__(self):
        return 0

    def sample(self, *args, **kwargs):
        raise RuntimeError('a frozen snapshot has no buffer to sample from')


class ReplayMemory(object):
    ''' DQN replay memory with the API of rlcard's Memory
    (save / sample / checkpoint_attributes / from_checkpoint), on a Ring.

    sample() returns (states, actions, rewards, next_states, dones,
    legal_masks): legal_masks is a (batch, num_actions) bool array. '''

    def __init__(self, memory_size, batch_size, num_actions=4):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.ring = Ring(memory_size)

    @property
    def memory(self):
        ''' Kept so `len(memory.memory)` keeps working. '''
        return range(len(self.ring))

    def __len__(self):
        return len(self.ring)

    def save(self, state, action, reward, next_state, legal_actions, done):
        if self.ring.data is None:
            shape = np.asarray(state).shape
            self.ring._make({'state': (shape, np.float16), 'next_state': (shape, np.float16),
                             'action': ((), np.int64), 'reward': ((), np.float32),
                             'done': ((), np.bool_), 'legal': ((self.num_actions,), np.bool_)})
        legal = np.zeros(self.num_actions, dtype=bool)
        legal[list(legal_actions)] = True
        self.ring.add({'state': state, 'next_state': next_state, 'action': action,
                       'reward': reward, 'done': done, 'legal': legal})

    def sample(self):
        b = self.ring.sample(self.batch_size)
        return (b['state'].astype(np.float32), b['action'], b['reward'],
                b['next_state'].astype(np.float32), b['done'], b['legal'])

    def checkpoint_attributes(self, include_data=True):
        state = self.ring._state()
        if not include_data:
            state.update(size=0, arrays={})
        return {'memory_size': self.memory_size, 'batch_size': self.batch_size,
                'num_actions': self.num_actions, 'format': 'arrays', 'ring': state,
                'next': self.ring.next if include_data else 0}

    @classmethod
    def from_checkpoint(cls, checkpoint):
        mem = cls(checkpoint['memory_size'], checkpoint['batch_size'], checkpoint.get('num_actions', 4))
        if checkpoint.get('format') == 'arrays':
            mem.ring._load(checkpoint['ring'])
            mem.ring.next = checkpoint.get('next', 0)
        else:
            # old list-of-Transition checkpoints
            for t in checkpoint.get('memory', []):
                mem.save(t.state, t.action, t.reward, t.next_state, t.legal_actions, t.done)
        return mem
