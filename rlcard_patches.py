''' Fixes for rlcard 1.2.0's limit hold'em game. Importing this module
applies them (once); functions.py imports it, so every training and
evaluation script gets them.

1. Game.init_game builds the first state of a hand *before* it resets
   `history_raise_nums`, so the small blind's first observation (raw_obs
   'raise_nums' and the raise-count bits of the 72-dim 'obs') carries the
   raise counts of the previous hand.
2. Game.step_back restores the raise history into a misspelled attribute
   (`history_raises_nums`), so after stepping back the observation keeps the
   raise counts of the branch that was undone. Deep CFR's tree traversal
   relies on step_back.
3. Speed: with allow_step_back, Game.step deep-copies the round, dealer and
   players before every step -- including their numpy RandomState objects,
   which are slow to copy. That was ~90% of Deep CFR's run time. The patched
   step stores only the mutable fields (a few numbers and short lists; the
   card objects are never mutated, so they are shared) and step_back writes
   them back into the same objects. Behaviour is identical (see
   test_exploitability.test_light_snapshot_matches_rlcard).
'''

from rlcard.games.limitholdem.game import LimitHoldemGame

if not getattr(LimitHoldemGame, '_thesis_patched', False):
    _original_init_game = LimitHoldemGame.init_game
    _original_step = LimitHoldemGame.step
    _original_step_back = LimitHoldemGame.step_back

    def init_game(self):
        # Reset before the original builds the first state from it; the
        # original then swaps in its own fresh list for the rest of the hand.
        self.history_raise_nums = [0 for _ in range(4)]
        return _original_init_game(self)

    def _snapshot(self):
        r = self.round
        return (r.game_pointer, r.raise_amount, r.have_raised, r.not_raise_num, list(r.raised),
                r.player_folded,
                self.game_pointer, self.round_counter,
                list(self.dealer.deck), self.dealer.pot, list(self.public_cards),
                [(list(p.hand), p.status, p.in_chips) for p in self.players],
                list(self.history_raise_nums))

    def _restore(self, snap):
        (game_pointer, raise_amount, have_raised, not_raise_num, raised, player_folded,
         self.game_pointer, self.round_counter, deck, pot, public_cards, players,
         self.history_raise_nums) = snap
        r = self.round
        r.game_pointer, r.raise_amount, r.have_raised = game_pointer, raise_amount, have_raised
        r.not_raise_num, r.raised, r.player_folded = not_raise_num, raised, player_folded
        self.dealer.deck, self.dealer.pot = deck, pot
        self.public_cards = public_cards
        for player, (hand, status, in_chips) in zip(self.players, players):
            player.hand, player.status, player.in_chips = hand, status, in_chips

    def step(self, action):
        if not self.allow_step_back:
            return _original_step(self, action)
        self.history.append(_snapshot(self))
        self.allow_step_back = False          # the original would deep-copy again
        try:
            return _original_step(self, action)
        finally:
            self.allow_step_back = True

    def step_back(self):
        if len(self.history) > 0:
            _restore(self, self.history.pop())
            return True
        return False

    LimitHoldemGame.init_game = init_game
    LimitHoldemGame.step = step
    LimitHoldemGame.step_back = step_back
    LimitHoldemGame._thesis_patched = True
    LimitHoldemGame._rlcard_step = _original_step              # for the equivalence test
    LimitHoldemGame._rlcard_step_back = _original_step_back

# 4. the rich observation (state['obs_rich'] in every limit-holdem env, and the
#    'limit-holdem-rich' env); imported last because it wraps the patches above
import obs_encoding  # noqa: E402,F401
