''' Fixes for bugs in rlcard 1.2.0's limit hold'em game. Importing this
module applies them (once); functions.py imports it, so every training and
evaluation script gets them.

1. Game.init_game builds the first state of a hand *before* it resets
   `history_raise_nums`, so the small blind's first observation (raw_obs
   'raise_nums' and the raise-count bits of the 72-dim 'obs') carries the
   raise counts of the previous hand.
2. Game.step_back restores the raise history into a misspelled attribute
   (`history_raises_nums`), so after stepping back the observation keeps the
   raise counts of the branch that was undone. Deep CFR's tree traversal
   relies on step_back.
'''

from rlcard.games.limitholdem.game import LimitHoldemGame

if not getattr(LimitHoldemGame, '_thesis_patched', False):
    _original_init_game = LimitHoldemGame.init_game

    def init_game(self):
        # Reset before the original builds the first state from it; the
        # original then swaps in its own fresh list for the rest of the hand.
        self.history_raise_nums = [0 for _ in range(4)]
        return _original_init_game(self)

    def step_back(self):
        if len(self.history) > 0:
            self.round, self.game_pointer, self.round_counter, self.dealer, self.public_cards, \
                self.players, self.history_raise_nums = self.history.pop()
            return True
        return False

    LimitHoldemGame.init_game = init_game
    LimitHoldemGame.step_back = step_back
    LimitHoldemGame._thesis_patched = True
