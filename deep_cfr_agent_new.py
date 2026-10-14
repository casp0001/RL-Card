# Copyright 2019 DATA Lab at Texas A&M University All rights reserved.
# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

''' Implements Deep CFR Algorithm.

The implementation is derived from:
    https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/deep_cfr.py

We modify the structure for single player game and rlcard package, and fix some bugs for loss calculation.

See https://arxiv.org/abs/1811.00164.

The algorithm defines an `advantage` and `strategy` networks that compute
advantages used to do regret matching across information sets and to approximate
the strategy profiles of the game.    To train these networks a fixed ring buffer
(other data structures may be used) memory is used to accumulate samples to
train the networks.

This version is ported to PyTorch. The algorithm, memory structures, traversal
logic, regret matching, and loss formulations are unchanged from the TF2/TF1
versions. Networks are nn.Module MLPs (ReLU hidden layers, linear output),
optimizers are torch.optim.Adam, and updates use standard autograd/backward
instead of tf.GradientTape.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import collections
import random
import numpy as np
import torch
import torch.nn as nn

from rlcard.utils.utils import remove_illegal
sys.setrecursionlimit(10000000)

# Networks and tensors live on CPU by default to match the original eager
# behaviour; flip this to 'cuda' if a GPU is available and desired.
_DEVICE = torch.device('xpu' if torch.xpu.is_available() else 'cpu')

AdvantageMemory = collections.namedtuple(
    'AdvantageMemory', 'info_state iteration advantage action')

StrategyMemory = collections.namedtuple(
    'StrategyMemory', 'info_state iteration strategy_action_probs')

# The traversal relies on step_back, which rlcard 1.2.0 gets wrong for limit
# hold'em; see rlcard_patches.py.
import rlcard_patches  # noqa: F401


class ReservoirBuffer(object):
    ''' Fixed-size buffer filled by reservoir sampling.

    Once full, the n-th added element replaces a uniformly random slot with
    probability capacity / n, so the buffer always holds a uniform sample of
    everything ever added. Deep CFR needs this (rather than FIFO) so that the
    linear-CFR average over all iterations is preserved.
    '''
    def __init__(self, replay_buffer_capacity):
        ''' Initialize the buffer
        '''

        self._replay_buffer_capacity = replay_buffer_capacity
        self._data = []
        self._add_calls = 0

    def add(self, element):
        '''Adds `element` to the buffer using reservoir sampling.

        Args:
            element: data to be added to the buffer.
        '''
        if len(self._data) < self._replay_buffer_capacity:
            self._data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._replay_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples):
        ''' Returns `num_samples` uniformly sampled from the buffer.

        Args:
            num_samples (int): number of samples to draw.

        Returns:
            sample data (list): a list of random sampled elements of the buffer

        Raises:
            ValueError: If there are less than `num_samples` elements in the buffer
        '''
        if len(self._data) < num_samples:
            raise ValueError("{} elements could not be sampled from size {}".format(
                num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def clear(self):
        ''' Clear the buffer
        '''
        self._data = []
        self._add_calls = 0

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


def _build_mlp(input_dim, layer_sizes, num_actions):
    ''' Build an MLP matching the original Sonnet `snt.nets.MLP` behaviour.

    Sonnet's MLP applies a (ReLU) activation between hidden layers and leaves
    the final output layer linear. We replicate that: ReLU on every hidden
    layer, linear on the final `num_actions` output layer.

    Args:
        input_dim (int): flattened info-state dimensionality.
        layer_sizes (list[int]): hidden layer sizes.
        num_actions (int): size of the linear output layer.

    Returns:
        torch.nn.Module: the constructed network.
    '''
    layers = []
    prev = input_dim
    for size in layer_sizes:
        layers.append(nn.Linear(prev, size))
        layers.append(nn.ReLU())
        prev = size
    layers.append(nn.Linear(prev, num_actions))
    model = nn.Sequential(*layers)
    return model.to(_DEVICE)


class DeepCFR():
    ''' Implement the Deep CFR Algorithm.

    See https://arxiv.org/abs/1811.00164.

    Define all networks and sampling buffers/memories.    Derive losses & learning
    steps. Initialize the game state and algorithmic variables.

    Note: batch sizes default to `None` implying that training over the full
        dataset in memory is done by default.    To sample from the memories you
        may set these values to something less than the full capacity of the
        memory.
    '''

    def __init__(self,
             session=None,
             env=None,
             policy_network_layers=(32, 32),
             advantage_network_layers=(32, 32),
             num_traversals=10,
             num_step=40,
             learning_rate=1e-4,
             batch_size_advantage=16,
             batch_size_strategy=16,
             memory_capacity=int(1e7)):
        ''' Initialize the Deep CFR

        Args:
            session: Unused. Kept for backwards-compatible call signatures.
            env (rlcard.env): for training (traverse)
            policy_network_layers (list[int]): Layer sizes of strategy net MLP
            advantage_network_layers (list[int]): Layer sizes of advantage net MLP
            num_traversals (int): Number of traversals per step
            learning_rate (float): Learning rate
            batch_size_advantage (int or None): Batch size to sample from advantage
            memories
            batch_size_strategy (int or None): Batch size to sample from strategy
            memories
            memory_capacity (int): Number af samples that can be stored in memory
        '''
        self._env = env
        self._session = session  # unused; retained for API compatibility
        self._learning_rate = learning_rate
        self._policy_network_layers = list(policy_network_layers)
        self._advantage_network_layers = list(advantage_network_layers)
        self._batch_size_advantage = batch_size_advantage
        self._batch_size_strategy = batch_size_strategy
        self._num_players = env.num_players
        self._num_step = num_step
        self.advantage_losses = collections.defaultdict(list)
        self.traverse = []

        # get initial state and players
        init_state, _ = self._env.reset()

        self._embedding_size = init_state['obs'].shape
        self._num_traversals = num_traversals
        self._num_actions = self._env.num_actions
        self._iteration = 1
        self._traverse_count = 0

        # Flattened info-state dimensionality (replaces the placeholder + reshape).
        self._info_state_dim = int(np.prod(self._embedding_size))

        # Define strategy network, optimizer & memory.
        self._strategy_memories = ReservoirBuffer(memory_capacity)
        self._policy_network = _build_mlp(
            self._info_state_dim, self._policy_network_layers, self._num_actions)
        self._optimizer_policy = torch.optim.Adam(
            self._policy_network.parameters(), lr=learning_rate)

        # Define advantage networks, optimizers & memories. (One per player)
        self._advantage_memories = [
            ReservoirBuffer(memory_capacity) for _ in range(self._num_players)
        ]
        self._advantage_networks = [
            _build_mlp(self._info_state_dim,
                       self._advantage_network_layers, self._num_actions)
            for _ in range(self._num_players)
        ]
        self._optimizer_advantages = [
            torch.optim.Adam(
                self._advantage_networks[p].parameters(), lr=learning_rate)
            for p in range(self._num_players)
        ]

    def train(self):
        ''' Perform tree traversal and train the network

        Returns:
            policy_network (torch.nn.Module): the trained policy network
            average advantage loss (float): players average advantage loss
            policy loss (float): policy loss
        '''
        for p in range(self._num_players):
            for _ in range(self._num_traversals):
                # Fresh deal for every traversal. Player p may be either blind;
                # the traversal handles whichever player acts first.
                init_state, _ = self._env.reset()
                self._traverse_game_tree(init_state, p)

            # Re-initialize the traverser's advantage network and train from
            # scratch. The other player's network must be kept: it supplies
            # the opponent strategy in the next player's traversals.
            self.reinitialize_advantage_network(p)
            for _ in range(self._num_step):
                self.advantage_losses[p].append(self._learn_advantage_network(p))

            self._iteration += 1

        # Train policy network.
        for _ in range(self._num_step):
            policy_loss = self._learn_strategy_network()

        adv_loss = [self.advantage_losses[p][-1] for p in self.advantage_losses.keys() if self.advantage_losses[p][-1] is not None]
        avg_adv_loss = sum(adv_loss) / len(adv_loss)

        return self._policy_network, avg_adv_loss, policy_loss

    def eval_step(self, state):
        ''' Predict the action given state for evaluation

        args:
            state (dict): current state

        returns:
            action (int): an action id
        '''
        obs = state['obs']
        legal_actions = state['legal_actions']
        action_prob = self.action_probabilities(obs)
        action_prob = remove_illegal(action_prob, legal_actions)
        action_prob /= action_prob.sum()
        action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
        return action

    def reinitialize_advantage_network(self, player):
        ''' Reinitialize one player's advantage network

        Rebuild the network (fresh random weights) and reset its optimizer
        state, reproducing the "train from scratch each iteration" behaviour.
        '''
        self._advantage_networks[player] = _build_mlp(
            self._info_state_dim,
            self._advantage_network_layers, self._num_actions)
        self._optimizer_advantages[player] = torch.optim.Adam(
            self._advantage_networks[player].parameters(),
            lr=self._learning_rate)

    def reinitialize_advantage_networks(self):
        ''' Reinitialize all advantage networks
        '''
        for p in range(self._num_players):
            self.reinitialize_advantage_network(p)

    def action_advantage(self, state, player):
        ''' Returns action advantages for a single batch.
        '''
        state = state['obs'].flatten()
        state = np.expand_dims(state, axis=0).astype(np.float32)
        with torch.no_grad():
            inp = torch.from_numpy(state).to(_DEVICE)
            advantages = self._advantage_networks[player](inp).cpu().numpy()[0]
        advantages = np.array([max(0., advantage) for advantage in advantages])
        return advantages

    def simulate_other(self, player, state):
        ''' Simulate the action for other players

        args:
            player (int): an player id
            state (dict): current state

        returns:
            action (int): an action id
        '''
        _, strategy = self._sample_action_from_advantage(state, player)
        # Recompute distribution dor numerical errors.
        probs = np.array(strategy)
        probs /= probs.sum()
        action = np.random.choice(range(self._num_actions), p=probs)
        return action

    def action_probabilities(self, state):
        ''' Returns action probabilites dict for a single batch.
        '''
        info_state_vector = np.array(state, dtype=np.float32)
        if len(info_state_vector.shape) == 1:
            info_state_vector = np.expand_dims(info_state_vector, axis=0)
        # Flatten to match the network's expected input dimensionality.
        info_state_vector = info_state_vector.reshape(
            info_state_vector.shape[0], -1)

        with torch.no_grad():
            inp = torch.from_numpy(info_state_vector).to(_DEVICE)
            logits = self._policy_network(inp)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # No rounding: rounding to 4 decimals could zero out every legal
        # action and make the caller's renormalisation divide by zero.
        return probs[0].astype(np.float64)

    def _traverse_game_tree(self, state, player, count = 0):
        ''' Performs a traversal of the game tree.

        Over a traversal the advantage and strategy memories are populated with
        computed advantage values and matched regrets respectively.

        Args:
            state (dict): Current rlcard game state.
            player (int): Player index for this traversal.

        Every env.step below is undone by exactly one env.step_back once its
        subtree has been evaluated, so the env is back at `state` when this
        call returns.

        Returns:
            payoff (np.ndarray): Expected payoff for each player.
        '''
        expected_payoff = collections.defaultdict(float)
        current_player = self._env.get_player_id()
        actions = state['legal_actions']
        if self._env.is_over():
            # Terminal state get returns.
            return np.asarray(self._env.get_payoffs())

        if current_player == player:
            sampled_regret = collections.defaultdict(float)
            # Update the policy over the info set & actions via regret matching.
            _, strategy = self._sample_action_from_advantage(state, player)
            for action in actions:
                child_state, _ = self._env.step(action)
                expected_payoff[action] = self._traverse_game_tree(child_state, player)
                self._env.step_back()

            for action in actions:
                sampled_regret[action] = expected_payoff[action][player]
                for a_ in actions:
                    sampled_regret[action] -= strategy[a_] * expected_payoff[a_][player]
            for act in actions:
                self._advantage_memories[player].add(AdvantageMemory(state['obs'].flatten(), self._iteration, sampled_regret[act], act))
            ev = np.zeros(self._env.num_players)
            for action in actions:
                ev += strategy[action] * np.asarray(expected_payoff[action])
            return ev
        else:
            other_player = current_player
            _, strategy = self._sample_action_from_advantage(state, other_player)
            # Recompute distribution dor numerical errors.
            probs = np.array(strategy)
            probs /= probs.sum()
            action = np.random.choice(range(self._num_actions), p=probs)
            child_state, _ = self._env.step(action)
            self._strategy_memories.add(
                StrategyMemory(
                    state['obs'].flatten(),
                    self._iteration, strategy))
            payoff = self._traverse_game_tree(child_state, player)
            self._env.step_back()
            return payoff

    def _sample_action_from_advantage(self, state, player):
        ''' Returns an info state policy by applying regret-matching.

        Args:
            state (dict): Current state.
            player (int): Player index over which to compute regrets.

        Returns:
            1. (list) Advantage values for info state actions indexed by action.
            2. (list) Matched regrets, prob for actions indexed by action.
        '''
        info_state = state['obs'].flatten()
        legal_actions = state['legal_actions']
        info_state = np.expand_dims(info_state, axis=0).astype(np.float32)
        with torch.no_grad():
            inp = torch.from_numpy(info_state).to(_DEVICE)
            advantages = self._advantage_networks[player](inp).cpu().numpy()[0]
        raw_advantages = advantages
        advantages = [max(0., advantage) for advantage in advantages]
        cumulative_regret = np.sum([advantages[action] for action in legal_actions])
        matched_regrets = np.array([0.] * self._num_actions)
        if cumulative_regret > 0.:
            for action in legal_actions:
                matched_regrets[action] = advantages[action] / cumulative_regret
        else:
            # No positive regret: play the legal action with the highest
            # predicted advantage, as in Brown et al. (2019). (Uniform play
            # here was ~50% more exploitable in their ablation.)
            best = max(legal_actions, key=lambda a: raw_advantages[a])
            matched_regrets[best] = 1.
        return advantages, matched_regrets

    @staticmethod
    def _flatten_state(state):
        ''' Flatten the given state represenatation

        Retained for API compatibility. Flattening is done with numpy/`reshape`
        at call sites rather than on a symbolic placeholder.

        Args:
            state: current state array

        Returns:
            flattened state array
        '''
        state = np.asarray(state)
        return state.reshape(state.shape[0], -1)

    def _learn_advantage_network(self, player):
        '''Compute the loss on sampled transitions and perform a Q-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Args:
            player (int): player id

        Returns:
            loss advantages (float): The average loss over the advantage network.
        '''
        if self._batch_size_advantage and self._batch_size_advantage < len(self._advantage_memories[player]._data):
            samples = self._advantage_memories[player].sample(self._batch_size_advantage)
        else:
            samples = self._advantage_memories[player]
        info_states = []
        advantages = []
        iterations = []
        acts = []
        for s in samples:
            info_states.append(s.info_state)
            advantages.append(s.advantage)
            iterations.append(s.iteration)
            acts.append(s.action)
        if info_states == []:
            return None
        # Ensure some samples have been gathered.
        info_states = np.array(info_states, dtype=np.float32)
        advantages = np.array(advantages, dtype=np.float32)
        iterations = np.array(iterations, dtype=np.float32)
        acts = np.array(acts, dtype=np.int64)

        network = self._advantage_networks[player]
        optimizer = self._optimizer_advantages[player]

        info_states_t = torch.from_numpy(info_states).to(_DEVICE)
        advantages_t = torch.from_numpy(advantages).to(_DEVICE)
        iterations_t = torch.from_numpy(iterations).to(_DEVICE)
        acts_t = torch.from_numpy(acts).to(_DEVICE)

        optimizer.zero_grad()
        preds = network(info_states_t)
        sqrt_iter = torch.sqrt(iterations_t / float(self._iteration))
        # Label: sqrt(iter) * advantage   -> shape [batch]
        lbl = sqrt_iter * advantages_t
        # Prediction: sqrt(iter) * net_output -> shape [batch, num_actions]
        pred = preds * sqrt_iter[:, None]

        batch_size = info_states_t.shape[0]
        gather_indices = torch.arange(batch_size, device=_DEVICE) * pred.shape[1] + acts_t
        action_predictions = pred.reshape(-1)[gather_indices]

        loss_advantages = torch.mean(torch.square(lbl - action_predictions))

        loss_advantages.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        return float(loss_advantages.item())

    def _learn_strategy_network(self):
        ''' compute the loss over the strategy network.

        Returns:
            The average loss obtained on this batch of transitions or `None`.
        '''
        if self._batch_size_strategy and self._batch_size_strategy < len(self._strategy_memories._data):
            samples = self._strategy_memories.sample(self._batch_size_strategy)
        else:
            samples = self._strategy_memories
        info_states = []
        action_probs = []
        iterations = []
        for s in samples:
            info_states.append(s.info_state)
            action_probs.append(s.strategy_action_probs)
            iterations.append(s.iteration)
        if info_states == []:
            return None
        info_states = np.array(info_states, dtype=np.float32)
        action_probs = np.array(np.squeeze(action_probs), dtype=np.float32)
        iterations = np.array(iterations, dtype=np.float32)

        info_states_t = torch.from_numpy(info_states).to(_DEVICE)
        action_probs_t = torch.from_numpy(action_probs).to(_DEVICE)
        iterations_t = torch.from_numpy(iterations).to(_DEVICE)

        self._optimizer_policy.zero_grad()
        logits = self._policy_network(info_states_t)
        action_probs_pred = torch.softmax(logits, dim=-1)
        sqrt_iter = torch.sqrt(iterations_t / float(self._iteration))[:, None]
        labels = action_probs_t * sqrt_iter
        predictions = action_probs_pred * sqrt_iter
        loss_strategy = torch.mean(torch.square(labels - predictions))

        loss_strategy.backward()
        torch.nn.utils.clip_grad_norm_(self._policy_network.parameters(), 1.0)
        self._optimizer_policy.step()
        return float(loss_strategy.item())
