''' A richer observation for limit hold'em ("rich", 153 values).

RLCard's default observation (72 values) is 52 card bits -- hole and board
cards mixed together -- plus the number of raises in each round. It cannot
tell the small blind from the big blind, a pocket pair from a pair on the
board, or who made the raises. The rich observation encodes what a real
player sees, from the acting player's point of view:

    offset  size  content
       0     52   own hole cards (one bit per card, RLCard's card2index order)
      52     52   board cards
     104      4   street (pre-flop, flop, turn, river), one-hot
     108      2   position: [small blind, big blind]
     110      1   facing a bet (something to call)
     111     20   raises in each round so far, one-hot 0..4 per round (as RLCard's)
     131      5   own raises in the current round, one-hot 0..4
     136      5   opponent's raises in the current round, one-hot 0..4
     141      1   already acted in the current round
     142      9   last raiser of each earlier round (pre-flop, flop, turn):
                  one-hot [nobody, me, opponent] per round, zeros if not played yet
     151      1   pot / 100 chips (the largest possible limit pot)
     152      1   amount to call / 4 chips (the largest possible call)

The hole cards come first so that exploitability.py's LBR can build the
observation of a hypothetical opponent hand by swapping those 52 bits, as it
does for the default encoding.

Importing this module (rlcard_patches.py does) makes every limit-holdem env
add both encodings to every state -- state['obs_rlcard'] (RLCard's 72 values)
and state['obs_rich'] -- and registers the env 'limit-holdem-rich', whose
state['obs'] is the rich observation (use it for training; in 'limit-holdem'
state['obs'] stays RLCard's). Agents trained on either encoding therefore play
correctly in any limit-holdem env when they read their own key (agent_loading
policies do).
'''

import os
import json

import numpy as np
import rlcard

# the light-snapshot step / step_back must be in place before they are wrapped here
import rlcard_patches  # noqa: F401
from rlcard.envs.limitholdem import LimitholdemEnv
from rlcard.games.limitholdem.game import LimitHoldemGame

RICH_DIM = 153
ENV_IDS = {'rlcard': 'limit-holdem', 'rich': 'limit-holdem-rich'}
ENCODINGS = tuple(ENV_IDS)

with open(os.path.join(rlcard.__path__[0], 'games/limitholdem/card2index.json')) as _f:
    CARD_INDEX = json.load(_f)

_STREET = {0: 0, 3: 1, 4: 2, 5: 3}


def encoding_of_dim(dim):
    ''' 'rich' or 'rlcard' from a network's input size. '''
    if dim == RICH_DIM:
        return 'rich'
    if dim == 72:
        return 'rlcard'
    raise ValueError(f'no observation encoding has {dim} values')


def obs_key(encoding):
    ''' The state key that holds an encoding's observation in ANY limit-holdem
    env (both keys are always present, whatever state['obs'] is). '''
    return 'obs_rich' if encoding == 'rich' else 'obs_rlcard'


def encode_rich(game, raw):
    ''' The rich observation of `raw` (a LimitHoldemGame.get_state dict, which
    the patch below gives a 'player_id'). '''
    me = raw['player_id']
    obs = np.zeros(RICH_DIM)
    for c in raw['hand']:
        obs[CARD_INDEX[c]] = 1
    for c in raw['public_cards']:
        obs[52 + CARD_INDEX[c]] = 1
    obs[104 + _STREET[len(raw['public_cards'])]] = 1
    obs[108 + (0 if me == game.sb_seat else 1)] = 1
    to_call = max(raw['all_chips']) - raw['my_chips']
    obs[110] = 1.0 if to_call > 0 else 0.0
    for i, n in enumerate(raw['raise_nums']):
        obs[111 + i * 5 + n] = 1
    current = game.round_counter
    this_round = [(p, a) for r, p, a in game.action_history if r == current]
    mine = sum(1 for p, a in this_round if p == me and a == 'raise')
    theirs = sum(1 for p, a in this_round if p != me and a == 'raise')
    obs[131 + min(mine, 4)] = 1
    obs[136 + min(theirs, 4)] = 1
    obs[141] = 1.0 if any(p == me for p, _ in this_round) else 0.0
    for r in range(min(current, 3)):
        raisers = [p for rr, p, a in game.action_history if rr == r and a == 'raise']
        slot = 0 if not raisers else (1 if raisers[-1] == me else 2)
        obs[142 + 3 * r + slot] = 1
    obs[151] = sum(raw['all_chips']) / 100.0
    obs[152] = to_call / 4.0
    return obs


# --------------------------------------------------------------------------- #
# Patches: betting history, blind seat, player id, state['obs_rich']
# --------------------------------------------------------------------------- #

if not getattr(LimitHoldemGame, '_rich_obs_patched', False):
    _init_game = LimitHoldemGame.init_game
    _step = LimitHoldemGame.step
    _step_back = LimitHoldemGame.step_back
    _get_state = LimitHoldemGame.get_state
    _extract_state = LimitholdemEnv._extract_state

    def init_game(self):
        self.action_history = []       # (round, player, action) of every action this hand
        state, player = _init_game(self)
        self.sb_seat = next(i for i, p in enumerate(self.players) if p.in_chips == self.small_blind)
        return state, player

    def step(self, action):
        record = (self.round_counter, self.game_pointer, action)
        result = _step(self, action)
        self.action_history.append(record)
        return result

    def step_back(self):
        if _step_back(self):
            self.action_history.pop()
            return True
        return False

    def get_state(self, player):
        state = _get_state(self, player)
        state['player_id'] = player
        return state

    def extract_state(self, state):
        extracted = _extract_state(self, state)
        extracted['obs_rlcard'] = extracted['obs']
        extracted['obs_rich'] = encode_rich(self.game, state)
        return extracted

    LimitHoldemGame.init_game = init_game
    LimitHoldemGame.step = step
    LimitHoldemGame.step_back = step_back
    LimitHoldemGame.get_state = get_state
    LimitholdemEnv._extract_state = extract_state
    LimitHoldemGame._rich_obs_patched = True


class LimitholdemRichEnv(LimitholdemEnv):
    ''' limit-holdem with the rich observation as state['obs']. '''

    def __init__(self, config):
        super().__init__(config)
        self.name = 'limit-holdem-rich'
        self.state_shape = [[RICH_DIM] for _ in range(self.num_players)]

    def _extract_state(self, state):
        extracted = super()._extract_state(state)
        extracted['obs'] = extracted['obs_rich']
        return extracted


try:
    from rlcard.envs.registration import register
    register(env_id='limit-holdem-rich', entry_point='obs_encoding:LimitholdemRichEnv')
except ValueError:
    pass   # already registered
