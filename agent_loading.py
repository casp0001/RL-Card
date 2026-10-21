''' Load any agent of this repo as a frozen Policy with explicit action
probabilities.

A Policy plays through RLCard's interface (eval_step / step, use_raw False,
feed is a no-op, so it never learns) and exposes its action distribution:

    policy.probs(state)                 -> np.array(4), sums to 1 over legal actions
    policy.probs_batch_obs(obs, legal)  -> np.array(N, 4)   (only if policy.batched)

The neural agents (DQN, NFSP, Deep CFR, PPO) are "batched": their decision
depends only on the 72-dim observation, so the probabilities of all 1326
possible hole-card combinations can be computed in one forward pass (used by
LBR in exploitability.py). The rule-based agents go through probs(state).

Kinds:
    dqn, nfsp, deepcfr, ppo     need --ckpt (file, or a run directory)
    tag, nit, lag               EquityAgent presets (eval_panel.py)
    honest, station, maniac, random
'''

import os

import numpy as np
import torch

import rlcard_patches  # noqa: F401
from obs_encoding import encoding_of_dim, obs_key

NUM_ACTIONS = 4
CALL, RAISE, FOLD, CHECK = 0, 1, 2, 3

CHECKPOINT_KINDS = ('dqn', 'nfsp', 'deepcfr', 'ppo')
RULE_KINDS = ('tag', 'nit', 'lag', 'honest', 'station', 'maniac', 'random')
KINDS = CHECKPOINT_KINDS + RULE_KINDS


def _legal_mask(legal, n=1):
    mask = np.zeros((n, NUM_ACTIONS), dtype=bool)
    mask[:, list(legal)] = True
    return mask


def _masked_softmax(logits, legal):
    logits = np.asarray(logits, dtype=np.float64)
    logits = np.where(_legal_mask(legal, len(logits)), logits, -np.inf)
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def _renormalise(probs, legal):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.where(_legal_mask(legal, len(probs)), probs, 0.0)
    total = probs.sum(axis=1, keepdims=True)
    uniform = _legal_mask(legal, len(probs)) / len(legal)
    return np.where(total > 0, probs / np.where(total > 0, total, 1), uniform)


class Policy(object):
    ''' Base class: subclasses implement probs (and probs_batch_obs if batched).

    Neural policies know which observation encoding their network was trained
    on ('rlcard' or 'rich', from its input size) and read it from the matching
    state key (state['obs'] or state['obs_rich']), so they play correctly in
    any limit-holdem env. '''
    use_raw = False
    batched = False
    kind = None
    encoding = 'rlcard'
    obs_key = 'obs_rlcard'

    def _set_input_dim(self, dim):
        self.encoding = encoding_of_dim(int(dim))
        self.obs_key = obs_key(self.encoding)

    def probs(self, state):
        legal = list(state['legal_actions'])
        obs = np.asarray(state[self.obs_key], dtype=np.float32)[None]
        return self.probs_batch_obs(obs, legal)[0]

    def eval_step(self, state):
        p = np.asarray(self.probs(state), dtype=np.float64)
        p = p / p.sum()            # float32 network outputs may be off by ~1e-7
        action = int(np.random.choice(NUM_ACTIONS, p=p))
        return action, {'probs': {a: float(p[a]) for a in range(NUM_ACTIONS)}}

    def step(self, state):
        return self.eval_step(state)[0]

    def feed(self, ts):
        pass


# --------------------------------------------------------------------------- #
# Neural agents (batched)
# --------------------------------------------------------------------------- #

class DQNPolicy(Policy):
    ''' Greedy Q policy (what DQNAgent.eval_step plays). '''
    batched = True
    kind = 'dqn'

    def __init__(self, agent):
        self.agent = agent
        self._set_input_dim(np.prod(agent.q_estimator.state_shape))

    def probs_batch_obs(self, obs, legal):
        q = self.agent.q_estimator.predict_nograd(np.asarray(obs, dtype=np.float32))
        q = np.where(_legal_mask(legal, len(q)), q, -np.inf)
        out = np.zeros_like(q)
        out[np.arange(len(q)), q.argmax(axis=1)] = 1.0
        return out


class NFSPPolicy(Policy):
    ''' The average policy (what NFSP plays at evaluation). '''
    batched = True
    kind = 'nfsp'

    def __init__(self, agent):
        self.agent = agent
        agent.evaluate_with = 'average_policy'
        agent.policy_network.eval()
        self._set_input_dim(np.prod(agent._state_shape))

    def probs_batch_obs(self, obs, legal):
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(self.agent.device)
            p = np.exp(self.agent.policy_network(x).cpu().numpy())
        return _renormalise(p, legal)


class DeepCFRPolicy(Policy):
    ''' The average-strategy (policy) network. '''
    batched = True
    kind = 'deepcfr'

    def __init__(self, agent):
        self.agent = agent
        self._set_input_dim(agent._info_state_dim)

    def probs_batch_obs(self, obs, legal):
        from deep_cfr_agent_new import _DEVICE
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(_DEVICE)
            logits = self.agent._policy_network(x).cpu().numpy()
        # softmax over all actions, then restrict to the legal ones (as eval does)
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return _renormalise(p / p.sum(axis=1, keepdims=True), legal)


class PPOPolicy(Policy):
    ''' Masked softmax policy (greedy if the agent evaluates greedily). '''
    batched = True
    kind = 'ppo'

    def __init__(self, agent):
        self.agent = agent
        agent.network.eval()
        self._set_input_dim(np.prod(agent.state_shape))

    def probs_batch_obs(self, obs, legal):
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(self.agent.device)
            logits, _ = self.agent.network(x)
        p = _masked_softmax(logits.cpu().numpy(), legal)
        if self.agent.evaluate_with == 'greedy':
            out = np.zeros_like(p)
            out[np.arange(len(p)), p.argmax(axis=1)] = 1.0
            return out
        return p


# --------------------------------------------------------------------------- #
# Rule-based agents
# --------------------------------------------------------------------------- #

class RandomPolicy(Policy):
    kind = 'random'

    def probs(self, state):
        legal = list(state['legal_actions'])
        p = np.zeros(NUM_ACTIONS)
        p[legal] = 1.0 / len(legal)
        return p


class DeterministicPolicy(Policy):
    ''' Wraps an agent whose step(state) is deterministic (station, maniac, honest). '''

    def __init__(self, agent, kind):
        self.agent = agent
        self.kind = kind

    def probs(self, state):
        p = np.zeros(NUM_ACTIONS)
        p[int(self.agent.step(state))] = 1.0
        return p


class HonestPolicy(Policy):
    ''' HonestAgent's deterministic rule, with its post-flop hand strength
    computed by the fast evaluator (same hand categories as HonestAgent's own
    best-of-21 evaluator, which is too slow to query for ~1000 hypothetical
    hands per decision). '''
    kind = 'honest'

    def __init__(self):
        from honest_agent_limited import HonestAgent
        self.agent = HonestAgent(num_actions=NUM_ACTIONS)

    @staticmethod
    def strength(raw_obs):
        from honest_agent_limited import hand_strength
        from equity_agent import evaluate, card_from_str
        if not raw_obs.get('public_cards'):
            return hand_strength(raw_obs)
        cards = [card_from_str(c) for c in raw_obs['hand'] + raw_obs['public_cards']]
        return round((evaluate(cards) >> 20) / 8.0, 4)   # category 0..8 -> HonestAgent's (cat-1)/8

    def probs(self, state):
        from honest_agent_limited import _choose_action
        p = np.zeros(NUM_ACTIONS)
        p[_choose_action(self.strength(state['raw_obs']), state['legal_actions'])] = 1.0
        return p


class EquityPolicy(Policy):
    ''' EquityAgent (tag / nit / lag) with its exact mixed action probabilities. '''

    def __init__(self, agent, kind):
        self.agent = agent
        self.kind = kind

    def probs(self, state):
        return np.asarray(self.agent.action_probs(state['raw_obs'], state.get('action_record', [])))

    def seed(self, seed):
        self.agent.seed(seed)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _resolve(path, filename):
    if path is None:
        raise ValueError('this kind needs --ckpt')
    return os.path.join(path, filename) if os.path.isdir(path) else path


def _load(path, device=None):
    ''' Load a checkpoint with every stored 'device' replaced by `device`
    (default cpu): from_checkpoint builds the networks on the stored device,
    so a checkpoint saved on a GPU would otherwise not load without one. '''
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    target = torch.device(device or 'cpu')

    def retarget(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == 'device' and (value is None or isinstance(value, (str, torch.device))):
                    obj[key] = target
                else:
                    retarget(value)
    retarget(checkpoint)
    return checkpoint


def _empty_memory(memory):
    ''' A DQN replay-memory checkpoint entry without its data (old or new format). '''
    return {'memory_size': memory['memory_size'], 'batch_size': memory['batch_size'],
            'num_actions': memory.get('num_actions', NUM_ACTIONS), 'memory': []}


def _hidden_layers(state_dict):
    ''' Hidden layer sizes of an nn.Sequential MLP from its weight shapes. '''
    weights = [v for k, v in state_dict.items() if k.endswith('weight') and v.dim() == 2]
    return [int(w.shape[0]) for w in weights[:-1]]


def load_policy(kind, ckpt=None, device=None):
    ''' Load an agent of the given kind as a frozen Policy. '''
    if kind == 'dqn':
        from dqn_agent_XPU import DQNAgent
        checkpoint = _load(_resolve(ckpt, 'checkpoint_dqn.pt'), device)
        checkpoint['memory'] = _empty_memory(checkpoint['memory'])   # the replay buffer is not needed
        agent = DQNAgent.from_checkpoint(checkpoint)
        return DQNPolicy(agent)
    if kind == 'nfsp':
        from nfsp_agent_XPU import NFSPAgent
        checkpoint = _load(_resolve(ckpt, 'checkpoint_nfsp.pt'), device)
        res = checkpoint['reservoir_buffer']               # buffers are not needed to play
        checkpoint['reservoir_buffer'] = {'reservoir_buffer_capacity': res['reservoir_buffer_capacity'],
                                          'data': [], 'add_calls': 0}
        checkpoint['rl_agent']['memory'] = _empty_memory(checkpoint['rl_agent']['memory'])
        agent = NFSPAgent.from_checkpoint(checkpoint)
        if device:
            agent.set_device(device)
        return NFSPPolicy(agent)
    if kind == 'deepcfr':
        import rlcard
        from deep_cfr_agent_new import DeepCFR
        checkpoint = _load(_resolve(ckpt, 'model.ckpt'))
        from obs_encoding import ENV_IDS
        input_dim = next(v for k, v in checkpoint['policy_network'].items() if k.endswith('weight')).shape[1]
        env = rlcard.make(ENV_IDS[encoding_of_dim(input_dim)], config={'allow_step_back': True})
        # layer sizes are read from the weights, so no --policy_layers needed
        agent = DeepCFR(env=env, memory_capacity=1,
                        policy_network_layers=_hidden_layers(checkpoint['policy_network']),
                        advantage_network_layers=_hidden_layers(checkpoint['advantage_network_0']))
        agent._policy_network.load_state_dict(checkpoint['policy_network'])
        agent._policy_network.eval()
        return DeepCFRPolicy(agent)
    if kind == 'ppo':
        from ppo_agent import PPOAgent
        agent = PPOAgent.from_checkpoint(_load(_resolve(ckpt, 'checkpoint_ppo.pt'), device))
        if device:
            agent.set_device(device)
        return PPOPolicy(agent)
    if kind in ('tag', 'nit', 'lag'):
        from equity_agent import EquityAgent
        from eval_panel import NIT_PARAMS, LAG_PARAMS
        params = {'tag': {}, 'nit': NIT_PARAMS, 'lag': LAG_PARAMS}[kind]
        return EquityPolicy(EquityAgent(seed=0, **params), kind)
    if kind == 'honest':
        return HonestPolicy()
    if kind in ('station', 'maniac'):
        from eval_panel import FixedPolicyAgent
        order = [CHECK, CALL] if kind == 'station' else [RAISE, CALL, CHECK]
        return DeterministicPolicy(FixedPolicyAgent(order), kind)
    if kind == 'random':
        return RandomPolicy()
    raise ValueError(f'unknown kind {kind!r}; choose from {KINDS}')
