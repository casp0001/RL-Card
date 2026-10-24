''' Opponent selection for the self-play training loops (DQN, PPO, NFSP).

Plain self-play always plays the newest frozen snapshot of the learner. That
can lock into an exploitable style (e.g. a PPO agent that never folds: against
a copy of itself that raises with anything, calling down *is* right). Two
optional remedies, both off by default (= the original behaviour):

  snapshot pool   keep the last `pool_size` snapshots; each episode plays the
                  newest with probability `latest_frac`, otherwise a uniformly
                  drawn older one (the usual fix for self-play cycling, cf.
                  fictitious self-play / league training)
  warm-up         during the first `warmup_episodes` episodes, play a
                  rule-based bot with a probability that fades linearly and
                  continuously from `warmup_start_frac` to 0 (or stays at it
                  with warmup_fade=0), so the learner first meets an opponent
                  whose raises mean strength instead of an untrained,
                  near-random network

The warm-up bot must not be an evaluation-panel member, otherwise the panel
scores (and tuning, which selects on them) would reward exploiting a bot the
agent trained against:
  honest      HonestAgent (not in the default panel; a different bot family)
  equity_mix  a slightly randomised EquityAgent: its style is redrawn every
              episode somewhere between the panel's nit and lag presets and
              every parameter is jittered by up to +-12 %, so it is a family
              of reasonable players and never exactly a panel member (it is
              still a relative of nit/tag/lag; 'honest' is the strictly
              held-out alternative)

The opponent stream uses its own RNG (seeded from the training seed), so it
does not change the learner's random numbers.
'''

import random

from functions import frozen_copy

WARMUP_BOTS = ('honest', 'equity_mix')


def add_self_play_args(parser):
    ''' Shared command-line arguments (literal defaults, read by train_gui.py). '''
    parser.add_argument('--opponent_pool', type=int, default=1,
                        help='Snapshots kept for self-play (1 = always the newest one)')
    parser.add_argument('--pool_latest_frac', type=float, default=0.5,
                        help='With a pool: probability of playing the newest snapshot')
    parser.add_argument('--warmup_episodes', type=int, default=0,
                        help='Episodes at the start in which a rule-based bot may be the opponent (0 = off)')
    parser.add_argument('--warmup_bot', type=str, default='honest', choices=list(WARMUP_BOTS),
                        help='Warm-up opponent; never use a panel member')
    parser.add_argument('--warmup_fade', type=int, default=1,
                        help='1 = bot share fades linearly to 0 over the warm-up, 0 = constant share')
    parser.add_argument('--warmup_start_frac', type=float, default=1.0,
                        help='Share of games against the bot at episode 0 (it then fades to 0)')


JITTER = 0.12        # relative per-parameter noise of the equity_mix bot
BOUNDS = {'raise_threshold': (0.2, 0.9), 'reraise_step': (0.02, 0.2), 'realisation': (0.4, 1.3),
          'bluff_frequency': (0.0, 0.6), 'bluff_max_equity': (0.2, 0.5)}


def _equity_mix_params(rng):
    ''' A style between the panel's nit and lag presets (t ~ U(0, 1)), with
    every value jittered by up to +-JITTER (relative, and +-0.03 absolute for
    the bluff frequencies, which are 0 for the nit), so the bot is a family
    of reasonable players rather than any one panel member. '''
    from eval_panel import NIT_PARAMS, LAG_PARAMS

    def noisy(key, x):
        lo, hi = BOUNDS[key]
        x = x * (1 + rng.uniform(-JITTER, JITTER)) + (rng.uniform(-0.03, 0.03) if key.startswith('bluff') else 0)
        return min(hi, max(lo, x))
    t = rng.random()
    params = {}
    for key in NIT_PARAMS:
        if key in LAG_PARAMS and key in BOUNDS:
            a, b = NIT_PARAMS[key], LAG_PARAMS[key]
            params[key] = ([noisy(key, x + t * (y - x)) for x, y in zip(a, b)] if isinstance(a, list)
                           else noisy(key, a + t * (b - a)))
    if 'bluff_max_equity' in LAG_PARAMS:
        from equity_agent import DEFAULT_PARAMS
        a = NIT_PARAMS.get('bluff_max_equity', DEFAULT_PARAMS['bluff_max_equity'])
        params['bluff_max_equity'] = noisy('bluff_max_equity', a + t * (LAG_PARAMS['bluff_max_equity'] - a))
    return params


class OpponentSchedule(object):
    ''' Picks the opponent of every training episode.

        schedule = OpponentSchedule(agent, args)
        for episode in ...:
            opponent = schedule.opponent(episode, agent)
    '''

    def __init__(self, agent, args):
        self.update_every = args.opponent_update_every
        self.pool_size = max(1, int(getattr(args, 'opponent_pool', 1)))
        self.latest_frac = float(getattr(args, 'pool_latest_frac', 0.5))
        self.warmup_episodes = int(getattr(args, 'warmup_episodes', 0))
        self.warmup_bot = getattr(args, 'warmup_bot', 'honest')
        self.warmup_fade = bool(int(getattr(args, 'warmup_fade', 1)))
        self.warmup_start_frac = float(getattr(args, 'warmup_start_frac', 1.0))
        if self.warmup_episodes and self.warmup_bot not in WARMUP_BOTS:
            raise ValueError(f'unknown warm-up bot {self.warmup_bot!r}; choose from {WARMUP_BOTS}')
        self.rng = random.Random(f'opponents-{args.seed}')
        self.pool = [frozen_copy(agent)]
        self._bot = None
        self.counts = {'latest': 0, 'older': 0, 'bot': 0}

    def bot_probability(self, episode):
        if episode >= self.warmup_episodes:
            return 0.0
        fade = 1.0 - episode / self.warmup_episodes if self.warmup_fade else 1.0
        return self.warmup_start_frac * fade

    def _bot_agent(self):
        if self.warmup_bot == 'honest':
            if self._bot is None:
                from honest_agent_limited import HonestAgent
                self._bot = HonestAgent(num_actions=4)
            return self._bot
        from equity_agent import EquityAgent
        if self._bot is None:
            self._bot = EquityAgent(seed=self.rng.randrange(2 ** 31))
        self._bot.params.update(_equity_mix_params(self.rng))
        return self._bot

    def opponent(self, episode, agent):
        ''' The opponent for this episode (takes a new snapshot when due). '''
        if episode > 0 and episode % self.update_every == 0:
            self.pool.append(frozen_copy(agent))
            del self.pool[:-self.pool_size]
        if self.warmup_episodes and self.rng.random() < self.bot_probability(episode):
            self.counts['bot'] += 1
            return self._bot_agent()
        if len(self.pool) == 1 or self.rng.random() < self.latest_frac:
            self.counts['latest'] += 1
            opponent = self.pool[-1]
        else:
            self.counts['older'] += 1
            opponent = self.pool[self.rng.randrange(len(self.pool) - 1)]
        if hasattr(opponent, 'sample_episode_policy'):
            # NFSP: resample the snapshot's mode (best response w.p. eta) every
            # episode, otherwise it keeps the mode it was copied in
            opponent.sample_episode_policy()
        return opponent
