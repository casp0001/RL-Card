''' PPO agent

Proximal Policy Optimization (Schulman et al., 2017 -- https://arxiv.org/abs/1707.06347)
with Generalized Advantage Estimation (Schulman et al., 2015 -- https://arxiv.org/abs/1506.02438)
and action masking, written for the RLCard imperfect-information card game suite.

The class deliberately mirrors the public surface of rlcard.agents.DQNAgent and
rlcard.agents.NFSPAgent:

    use_raw, step(state), eval_step(state), feed(ts), train(), set_device(device),
    checkpoint_attributes(), from_checkpoint(checkpoint), save_checkpoint(path)

so it is a drop-in replacement inside the standard RLCard training loop
(env.run -> reorganize -> agent.feed) and works unchanged with
rlcard.utils.tournament, Logger and plot_curve.

Two things differ from DQN/NFSP by necessity, because PPO is on-policy:

  * feed() buffers whole episodes and only performs an update once
    `update_every` transitions of *completed* episodes have accumulated. There
    is no replay memory -- the rollout buffer is cleared after every update.
  * step() caches the log-probability and state value of the action it sampled.
    feed() consumes that cache, so the importance ratio in the clipped
    objective is computed against the exact behaviour policy that generated the
    action, no matter when the update happens to fire.
'''

import collections

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = collections.namedtuple(
    'Transition',
    ['obs', 'action', 'reward', 'next_obs', 'done', 'legal_actions', 'log_prob', 'value'],
)

# Value used instead of -inf when masking illegal-action logits. -inf would make
# log_softmax produce NaN gradients for states in which an action is illegal.
NEG_INF = -1e8


class PPOAgent(object):
    ''' A PPO agent that plays any RLCard environment with a discrete,
        per-state-masked action space (limit Hold'em, Leduc Hold'em, ...).
    '''

    def __init__(self,
                 num_actions=4,
                 state_shape=None,
                 mlp_layers=None,
                 activation='relu',
                 learning_rate=3e-4,
                 discount_factor=0.99,
                 gae_lambda=0.95,
                 clip_range=0.2,
                 value_clip_range=None,
                 entropy_coef=0.01,
                 value_coef=0.5,
                 max_grad_norm=0.5,
                 update_every=2048,
                 num_epochs=4,
                 batch_size=256,
                 normalize_advantage=True,
                 reward_scaling=1.0,
                 target_kl=None,
                 evaluate_with='stochastic',
                 device=None,
                 save_path=None,
                 save_every=float('inf')):
        ''' Initialize the PPO agent.

        Args:
            num_actions (int): The number of actions in the environment.
            state_shape (list): The shape of the observation, e.g. [72] for limit Hold'em.
            mlp_layers (list): Hidden layer sizes of the shared actor-critic trunk.
            activation (str): 'relu' or 'tanh' activation for the trunk.
            learning_rate (float): Adam learning rate.
            discount_factor (float): Gamma. Poker hands are short and the only
              non-zero reward is terminal, so values close to 1.0 are sensible.
            gae_lambda (float): Lambda of the GAE(gamma, lambda) advantage estimator.
            clip_range (float): Epsilon of the clipped surrogate policy objective.
            value_clip_range (float or None): If set, the value loss is clipped
              to this range around the old value prediction. None disables it.
            entropy_coef (float): Weight of the entropy bonus. Keeping this
              above zero matters in poker: a deterministic policy is trivially
              exploitable, so the agent should retain a mixed strategy.
            value_coef (float): Weight of the value function loss.
            max_grad_norm (float or None): Global gradient-norm clip. None disables it.
            update_every (int): Number of buffered transitions (from completed
              episodes) that triggers a PPO update.
            num_epochs (int): Number of passes over the rollout buffer per update.
            batch_size (int): Minibatch size within an epoch.
            normalize_advantage (bool): Standardize advantages per update.
            reward_scaling (float): Multiplier applied to rewards before they
              enter the buffer. Purely internal -- it rescales the value-function
              targets and does not affect the payoffs reported by tournament().
            target_kl (float or None): If set, stop the epoch loop early once the
              approximate KL between the old and new policy exceeds this.
            evaluate_with (str): 'stochastic' samples from the policy at
              evaluation time (recommended for poker), 'greedy' takes the argmax.
            device (torch.device): cpu or gpu.
            save_path (str): Directory for periodic checkpoints.
            save_every (int): Save a checkpoint every X PPO updates.
        '''
        self.use_raw = False

        if mlp_layers is None:
            mlp_layers = [128, 128]

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_every = update_every
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.normalize_advantage = normalize_advantage
        self.reward_scaling = reward_scaling
        self.target_kl = target_kl

        if evaluate_with not in ('stochastic', 'greedy'):
            raise ValueError("'evaluate_with' should be either 'stochastic' or 'greedy'.")
        self.evaluate_with = evaluate_with

        # Torch device
        if device is None:
            self.device = torch.device(os.environ.get('RL_DEVICE', 'cpu'))
        else:
            self.device = device

        # Total timesteps fed to the agent
        self.total_t = 0

        # Total number of PPO updates performed
        self.train_t = 0

        # Actor-critic network (shared trunk, separate policy and value heads)
        self.network = ActorCriticNetwork(
            num_actions=num_actions,
            state_shape=state_shape,
            mlp_layers=mlp_layers,
            activation=activation,
        ).to(self.device)
        self.network.eval()

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

        # On-policy rollout buffer
        self.memory = RolloutBuffer()

        # Behaviour-policy cache filled by step() and drained by feed(); see the
        # module docstring. Bounded so that using this agent as a frozen,
        # never-fed self-play opponent cannot grow it without limit.
        self._pending = collections.deque(maxlen=1024)

        # Diagnostics from the most recent update, handy for logging
        self.last_loss = None

        # Checkpoint saving parameters
        self.save_path = save_path
        self.save_every = save_every

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #

    def step(self, state):
        ''' Sample an action from the current policy, for generating training data.

        Args:
            state (dict): The current RLCard state.

        Returns:
            action (int): An action id.
        '''
        obs = np.asarray(state['obs'], dtype=np.float32)
        legal_actions = list(state['legal_actions'].keys())

        log_probs, value = self._policy_nograd(obs, legal_actions)
        probs = np.exp(log_probs)
        # Guard against numerical drift so np.random.choice never rejects p
        probs = probs / probs.sum()
        action = int(np.random.choice(self.num_actions, p=probs))

        self._pending.append((obs, action, float(log_probs[action]), float(value)))

        return action

    def eval_step(self, state):
        ''' Predict the action for evaluation purposes.

        Args:
            state (dict): The current RLCard state.

        Returns:
            action (int): An action id.
            info (dict): A dictionary containing information.
        '''
        obs = np.asarray(state['obs'], dtype=np.float32)
        legal_actions = list(state['legal_actions'].keys())

        log_probs, value = self._policy_nograd(obs, legal_actions)
        probs = np.exp(log_probs)
        probs = probs / probs.sum()

        if self.evaluate_with == 'greedy':
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(self.num_actions, p=probs))

        info = {}
        info['probs'] = {
            state['raw_legal_actions'][i]: float(probs[legal_actions[i]])
            for i in range(len(legal_actions))
        }
        info['state_value'] = float(value)

        return action, info

    def sample_episode_policy(self):
        ''' No-op, kept so that PPOAgent can be dropped into training loops
            written for NFSPAgent (which mixes best-response and average policy
            per episode). PPO always acts with its single stochastic policy.
        '''
        pass

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def feed(self, ts):
        ''' Store one transition in the rollout buffer, and run a PPO update
            once enough completed episodes have accumulated.

        Args:
            ts (list): a list of 5 elements that represent the transition,
              as produced by rlcard.utils.reorganize:
              (state, action, reward, next_state, done)
        '''
        (state, action, reward, next_state, done) = tuple(ts)

        obs = np.asarray(state['obs'], dtype=np.float32)
        next_obs = np.asarray(next_state['obs'], dtype=np.float32)
        legal_actions = list(state['legal_actions'].keys())

        cached = self._pop_pending(obs, action)
        if cached is None:
            # The action was not produced by this object's step() (for example
            # it was imitated, or replayed). Fall back to scoring it under the
            # current policy, which is exact as long as the network has not been
            # updated since the action was taken.
            log_probs, value = self._policy_nograd(obs, legal_actions)
            log_prob, value = float(log_probs[action]), float(value)
        else:
            log_prob, value = cached

        self.memory.add(Transition(
            obs=obs,
            action=action,
            reward=reward * self.reward_scaling,
            next_obs=next_obs,
            done=done,
            legal_actions=legal_actions,
            log_prob=log_prob,
            value=value,
        ))

        self.total_t += 1

        if self.memory.size >= self.update_every:
            self.train()

    def train(self):
        ''' Run one PPO update over the buffered rollout, then clear the buffer.

        Returns:
            loss (float): The mean total loss over the update, or None if there
              was nothing complete enough to learn from.
        '''
        batch = self._prepare_batch()
        if batch is None:
            return None

        obs, actions, old_log_probs, old_values, advantages, returns, masks = batch
        num_samples = obs.shape[0]
        batch_size = min(self.batch_size, num_samples)
        indices = np.arange(num_samples)

        self.network.train()

        total_loss, pg_loss_sum, vf_loss_sum, ent_sum, n_batches = 0.0, 0.0, 0.0, 0.0, 0
        approx_kl = 0.0
        stop_early = False

        for _ in range(self.num_epochs):
            np.random.shuffle(indices)
            # Updates fire at episode ends, so num_samples is update_every plus
            # a few; a ragged tail shorter than half a minibatch would take a
            # full-size gradient step on a handful of samples, so it joins the
            # previous minibatch instead.
            starts = list(range(0, num_samples, batch_size))
            if len(starts) > 1 and num_samples - starts[-1] < batch_size // 2:
                starts.pop()
            for k, start in enumerate(starts):
                end = starts[k + 1] if k + 1 < len(starts) else num_samples
                mb = indices[start:end]
                if len(mb) < 2:
                    continue

                mb_obs = obs[mb]
                mb_actions = actions[mb]
                mb_old_log_probs = old_log_probs[mb]
                mb_old_values = old_values[mb]
                mb_advantages = advantages[mb]
                mb_returns = returns[mb]
                mb_masks = masks[mb]

                logits, values = self.network(mb_obs)
                logits = logits.masked_fill(~mb_masks, NEG_INF)
                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()

                new_log_probs = log_probs.gather(1, mb_actions.unsqueeze(-1)).squeeze(-1)
                entropy = -(probs * log_probs).sum(dim=-1)

                # Clipped surrogate policy objective
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss, optionally clipped around the old prediction
                if self.value_clip_range is None:
                    value_loss = F.mse_loss(values, mb_returns)
                else:
                    clipped_values = mb_old_values + torch.clamp(
                        values - mb_old_values, -self.value_clip_range, self.value_clip_range)
                    value_loss = torch.max(
                        (values - mb_returns) ** 2,
                        (clipped_values - mb_returns) ** 2,
                    ).mean()

                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                pg_loss_sum += policy_loss.item()
                vf_loss_sum += value_loss.item()
                ent_sum += entropy.mean().item()
                n_batches += 1

                # Schulman's low-variance estimator of KL(old || new)
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()

            if self.target_kl is not None and approx_kl > self.target_kl:
                stop_early = True
                break

        self.network.eval()
        self.memory.clear()

        if n_batches == 0:
            return None

        mean_loss = total_loss / n_batches
        self.last_loss = {
            'loss': mean_loss,
            'policy_loss': pg_loss_sum / n_batches,
            'value_loss': vf_loss_sum / n_batches,
            'entropy': ent_sum / n_batches,
            'approx_kl': approx_kl,
            'early_stop': stop_early,
        }

        print('\rINFO - Step {}, ppo-loss: {:.5f}, policy: {:.5f}, value: {:.5f}, entropy: {:.4f}'.format(
            self.total_t, mean_loss, self.last_loss['policy_loss'],
            self.last_loss['value_loss'], self.last_loss['entropy']), end='')

        self.train_t += 1

        if self.save_path and self.train_t % self.save_every == 0:
            # To preserve every checkpoint separately,
            # add another argument to the function call parameterized by self.train_t
            self.save_checkpoint(self.save_path)
            print("\nINFO - Saved model checkpoint.")

        return mean_loss

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _masked_logits(self, logits, legal_actions):
        ''' Set the logits of illegal actions to a large negative constant.

        Args:
            logits (Tensor): (batch, num_actions)
            legal_actions (list): indices of the legal actions

        Returns:
            (Tensor): (batch, num_actions) masked logits
        '''
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, legal_actions] = True
        return logits.masked_fill(~mask, NEG_INF)

    def _policy_nograd(self, obs, legal_actions):
        ''' Score a single observation without touching the computation graph.

        Args:
            obs (numpy.array): the observation
            legal_actions (list): indices of the legal actions

        Returns:
            log_probs (numpy.array): (num_actions,) masked log-probabilities
            value (float): the state value
        '''
        with torch.no_grad():
            s = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            logits, value = self.network(s)
            log_probs = F.log_softmax(self._masked_logits(logits, legal_actions), dim=-1)
        return log_probs[0].cpu().numpy(), value[0].item()

    def _pop_pending(self, obs, action):
        ''' Drain the behaviour-policy cache up to the entry matching this
            (obs, action) pair and return it.

        Entries that do not match are transitions this object acted on but was
        never fed (a self-play seat whose trajectory is discarded, say), so they
        are dropped.

        Returns:
            (tuple) (log_prob, value), or None if no entry matched.
        '''
        while self._pending:
            p_obs, p_action, log_prob, value = self._pending.popleft()
            if p_action == action and np.array_equal(p_obs, obs):
                return log_prob, value
        return None

    def _prepare_batch(self):
        ''' Turn the buffered episodes into flat PPO training tensors, computing
            GAE advantages and value targets episode by episode.

        Returns:
            (tuple) or None if the buffer holds no completed episode.
        '''
        episodes = self.memory.episodes
        if not episodes:
            return None

        obs_list, action_list, log_prob_list, value_list = [], [], [], []
        advantage_list, return_list, mask_list = [], [], []

        for episode in episodes:
            rewards = np.array([t.reward for t in episode], dtype=np.float32)
            values = np.array([t.value for t in episode], dtype=np.float32)
            dones = np.array([t.done for t in episode], dtype=bool)
            length = len(episode)

            # Bootstrap value for a truncated episode. RLCard's reorganize()
            # always terminates a trajectory, so this normally stays 0.
            if dones[-1]:
                bootstrap = 0.0
            else:
                last = episode[-1]
                _, bootstrap = self._policy_nograd(last.next_obs, last.legal_actions)

            advantages = np.zeros(length, dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(length)):
                if t == length - 1:
                    next_non_terminal = 0.0 if dones[t] else 1.0
                    next_value = bootstrap
                else:
                    next_non_terminal = 1.0
                    next_value = values[t + 1]
                delta = rewards[t] + self.discount_factor * next_non_terminal * next_value - values[t]
                last_gae = delta + self.discount_factor * self.gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae

            returns = advantages + values

            for t, transition in enumerate(episode):
                obs_list.append(transition.obs)
                action_list.append(transition.action)
                log_prob_list.append(transition.log_prob)
                value_list.append(transition.value)
                advantage_list.append(advantages[t])
                return_list.append(returns[t])
                mask = np.zeros(self.num_actions, dtype=bool)
                mask[transition.legal_actions] = True
                mask_list.append(mask)

        advantages = np.array(advantage_list, dtype=np.float32)
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def to_tensor(array, dtype):
            return torch.as_tensor(np.array(array), dtype=dtype, device=self.device)

        return (
            to_tensor(obs_list, torch.float32),
            to_tensor(action_list, torch.long),
            to_tensor(log_prob_list, torch.float32),
            to_tensor(value_list, torch.float32),
            to_tensor(advantages, torch.float32),
            to_tensor(return_list, torch.float32),
            to_tensor(mask_list, torch.bool),
        )

    def set_device(self, device):
        self.device = device
        self.network.to(device)

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def checkpoint_attributes(self):
        '''
        Return the current checkpoint attributes (dict)
        Checkpoint attributes are used to save and restore the model in the middle of training
        Saves the model state dict, optimizer state dict, and all other instance variables
        '''
        return {
            'agent_type': 'PPOAgent',
            'network': self.network.checkpoint_attributes(),
            'optimizer': self.optimizer.state_dict(),
            'memory': self.memory.checkpoint_attributes(),
            'total_t': self.total_t,
            'train_t': self.train_t,
            'num_actions': self.num_actions,
            'state_shape': self.state_shape,
            'mlp_layers': self.mlp_layers,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'value_clip_range': self.value_clip_range,
            'entropy_coef': self.entropy_coef,
            'value_coef': self.value_coef,
            'max_grad_norm': self.max_grad_norm,
            'update_every': self.update_every,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'normalize_advantage': self.normalize_advantage,
            'reward_scaling': self.reward_scaling,
            'target_kl': self.target_kl,
            'evaluate_with': self.evaluate_with,
            'device': self.device,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        '''
        Restore the model from a checkpoint

        Args:
            checkpoint (dict): the checkpoint attributes generated by checkpoint_attributes()
        '''
        print("\nINFO - Restoring model from checkpoint...")
        agent_instance = cls(
            num_actions=checkpoint['num_actions'],
            state_shape=checkpoint['state_shape'],
            mlp_layers=checkpoint['mlp_layers'],
            activation=checkpoint['activation'],
            learning_rate=checkpoint['learning_rate'],
            discount_factor=checkpoint['discount_factor'],
            gae_lambda=checkpoint['gae_lambda'],
            clip_range=checkpoint['clip_range'],
            value_clip_range=checkpoint['value_clip_range'],
            entropy_coef=checkpoint['entropy_coef'],
            value_coef=checkpoint['value_coef'],
            max_grad_norm=checkpoint['max_grad_norm'],
            update_every=checkpoint['update_every'],
            num_epochs=checkpoint['num_epochs'],
            batch_size=checkpoint['batch_size'],
            normalize_advantage=checkpoint['normalize_advantage'],
            reward_scaling=checkpoint['reward_scaling'],
            target_kl=checkpoint['target_kl'],
            evaluate_with=checkpoint['evaluate_with'],
            device=checkpoint['device'],
        )

        agent_instance.total_t = checkpoint['total_t']
        agent_instance.train_t = checkpoint['train_t']

        agent_instance.network = ActorCriticNetwork.from_checkpoint(checkpoint['network'])
        agent_instance.network.to(agent_instance.device)
        agent_instance.network.eval()
        agent_instance.optimizer = torch.optim.Adam(
            agent_instance.network.parameters(), lr=agent_instance.learning_rate)
        agent_instance.optimizer.load_state_dict(checkpoint['optimizer'])
        agent_instance.memory = RolloutBuffer.from_checkpoint(checkpoint['memory'])

        return agent_instance

    def save_checkpoint(self, path, filename='checkpoint_ppo.pt'):
        ''' Save the model checkpoint (all attributes)

        Args:
            path (str): the path to save the model
        '''
        torch.save(self.checkpoint_attributes(), path + '/' + filename)


class ActorCriticNetwork(nn.Module):
    ''' Shared-trunk actor-critic network. Forward returns raw (unmasked)
        action logits and the state value.

        Note there is no BatchNorm here, unlike the RLCard DQN estimator. The
        policy is evaluated one state at a time while acting (eval mode, running
        statistics) but in batches while training (train mode, batch
        statistics); that mismatch would corrupt the PPO importance ratio, which
        assumes the same function produced the stored log-probabilities. The
        limit Hold'em observation is a 0/1 indicator vector anyway, so it needs
        no input normalization.
    '''

    def __init__(self, num_actions=4, state_shape=None, mlp_layers=None, activation='relu'):
        ''' Initialize the actor-critic network.

        Args:
            num_actions (int): number of output actions
            state_shape (list): shape of the state tensor
            mlp_layers (list): output size of each hidden layer of the trunk
            activation (str): 'relu' or 'tanh'
        '''
        super(ActorCriticNetwork, self).__init__()

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.activation = activation

        if activation == 'relu':
            act_layer = nn.ReLU
        elif activation == 'tanh':
            act_layer = nn.Tanh
        else:
            raise ValueError("'activation' should be either 'relu' or 'tanh'.")

        layer_dims = [int(np.prod(self.state_shape))] + list(self.mlp_layers)
        trunk = [nn.Flatten()]
        for i in range(len(layer_dims) - 1):
            trunk.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=True))
            trunk.append(act_layer())
        self.trunk = nn.Sequential(*trunk)

        self.actor_head = nn.Linear(layer_dims[-1], num_actions, bias=True)
        self.critic_head = nn.Linear(layer_dims[-1], 1, bias=True)

        # Orthogonal initialization with the gains PPO implementations converged
        # on: sqrt(2) in the trunk, 0.01 on the policy head so the initial
        # policy is close to uniform, 1.0 on the value head.
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)

    def forward(self, s):
        ''' Predict action logits and state value

        Args:
            s (Tensor): (batch, state_shape)

        Returns:
            logits (Tensor): (batch, num_actions) unmasked action logits
            value (Tensor): (batch,) state values
        '''
        h = self.trunk(s)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)

    def checkpoint_attributes(self):
        ''' Return the attributes needed to restore the model from a checkpoint
        '''
        return {
            'num_actions': self.num_actions,
            'state_shape': self.state_shape,
            'mlp_layers': self.mlp_layers,
            'activation': self.activation,
            'state_dict': self.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        ''' Restore the model from a checkpoint
        '''
        network = cls(
            num_actions=checkpoint['num_actions'],
            state_shape=checkpoint['state_shape'],
            mlp_layers=checkpoint['mlp_layers'],
            activation=checkpoint['activation'],
        )
        network.load_state_dict(checkpoint['state_dict'])
        return network


class RolloutBuffer(object):
    ''' On-policy buffer that groups transitions into whole episodes.

        PPO needs complete episodes so that GAE can be unrolled backwards from
        the terminal reward, which in poker is the only non-zero reward. A
        partially fed episode is held aside in `current` and only promoted to
        `episodes` once its terminal transition arrives.
    '''

    def __init__(self):
        self.episodes = []   # completed episodes, each a list of Transition
        self.current = []    # episode currently being fed
        self.size = 0        # number of transitions across completed episodes

    def add(self, transition):
        ''' Append a transition, closing off the episode when it terminates.

        Args:
            transition (Transition): the transition to store

        Returns:
            (bool): whether this transition completed an episode
        '''
        self.current.append(transition)
        if transition.done:
            self.episodes.append(self.current)
            self.size += len(self.current)
            self.current = []
            return True
        return False

    def clear(self):
        ''' Drop the completed episodes after an update. Any in-progress episode
            is kept, since its transitions are still being fed.
        '''
        self.episodes = []
        self.size = 0

    def checkpoint_attributes(self):
        ''' Returns the attributes that need to be checkpointed
        '''
        return {
            'episodes': self.episodes,
            'current': self.current,
            'size': self.size,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        '''
        Restores the attributes from the checkpoint

        Args:
            checkpoint (dict): the checkpoint dictionary

        Returns:
            instance (RolloutBuffer): the restored instance
        '''
        instance = cls()
        instance.episodes = checkpoint['episodes']
        instance.current = checkpoint['current']
        instance.size = checkpoint['size']
        return instance

    def __len__(self):
        return self.size
