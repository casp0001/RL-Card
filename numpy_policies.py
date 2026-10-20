''' Pure-numpy versions of the trained agents' policies (no torch needed).

export_for_web.py converts a trained agent (DQN, NFSP, Deep CFR, PPO) into a
.npz file holding its network as a list of simple operations, and checks that
this module reproduces the torch policy's action probabilities. The web app
(RL-Card-web) loads the files with NumpyPolicy and needs only numpy.

A policy file contains
    __meta__   JSON: name, kind, encoding ('rich' / 'rlcard'), output, ops
    op{i}_*    the arrays of operation i

Operations (applied in order to a (batch, input) array):
    batchnorm  (x - mean) / sqrt(var + eps) * weight + bias   (eval mode)
    linear     x @ W.T + b
    relu, tanh

Outputs (how the final values become action probabilities over the legal actions):
    greedy_q         DQN: all probability on the legal action with the largest Q-value
    log_probs        NFSP average policy: exp(), restricted to legal actions, renormalised
    softmax_renorm   Deep CFR: softmax over all actions, restricted to legal ones, renormalised
    masked_softmax   PPO: softmax over the legal actions' logits
    masked_greedy    PPO evaluated greedily: the legal action with the largest logit
'''

import json

import numpy as np

NUM_ACTIONS = 4


def obs_key(encoding):
    return 'obs_rich' if encoding == 'rich' else 'obs_rlcard'


class NumpyPolicy(object):

    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data['__meta__']))
        self.name = meta['name']
        self.kind = meta['kind']
        self.encoding = meta['encoding']
        self.output = meta['output']
        self.obs_key = obs_key(self.encoding)
        self.ops = []
        for i, op in enumerate(meta['ops']):
            arrays = {k: data[f'op{i}_{k}'].astype(np.float64) for k in op.get('arrays', [])}
            self.ops.append((op['type'], arrays, op.get('eps', 1e-5)))
        self.input_dim = meta['input_dim']

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
        for kind, a, eps in self.ops:
            if kind == 'batchnorm':
                x = (x - a['mean']) / np.sqrt(a['var'] + eps) * a['weight'] + a['bias']
            elif kind == 'linear':
                x = x @ a['W'].T + a['b']
            elif kind == 'relu':
                x = np.maximum(x, 0.0)
            elif kind == 'tanh':
                x = np.tanh(x)
            else:
                raise ValueError(f'unknown operation {kind!r}')
        return x

    def probs_batch_obs(self, obs, legal):
        out = self.forward(obs)
        legal = list(legal)
        mask = np.zeros((len(out), NUM_ACTIONS), dtype=bool)
        mask[:, legal] = True
        if self.output in ('greedy_q', 'masked_greedy'):
            best = np.where(mask, out, -np.inf).argmax(axis=1)
            p = np.zeros_like(out)
            p[np.arange(len(out)), best] = 1.0
            return p
        if self.output == 'masked_softmax':
            z = np.where(mask, out, -np.inf)
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)
        if self.output == 'log_probs':
            p = np.exp(out)
        elif self.output == 'softmax_renorm':
            z = out - out.max(axis=1, keepdims=True)
            p = np.exp(z)
            p = p / p.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f'unknown output {self.output!r}')
        p = np.where(mask, p, 0.0)
        total = p.sum(axis=1, keepdims=True)
        uniform = mask / len(legal)
        return np.where(total > 0, p / np.where(total > 0, total, 1), uniform)

    def probs(self, state):
        obs = np.asarray(state[self.obs_key], dtype=np.float64)[None]
        return self.probs_batch_obs(obs, state['legal_actions'])[0]
