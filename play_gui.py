''' Play limit hold'em against any agent of this repo.

    python play_gui.py

Pick an opponent (a trained checkpoint -- dqn / nfsp / deepcfr / ppo -- or a
rule-based bot), press "New session" and play. Keyboard: F fold, C check/call,
R bet/raise, N or Space next hand.

The game is RLCard's limit-holdem environment driven one step at a time
(env.run() cannot be used because it blocks waiting for the human). Blinds
are assigned at random every hand, exactly as in training. Every hand is
recorded with behaviour_tracker.HandRecorder and auto-saved to
experiments/human_sessions/<opponent>_<time>.jsonl, so sessions can be
reviewed here ("Review / replay...") or analysed with
    python behaviour_tracker.py stats <file> --role you

"Show agent's thinking" displays the agent's action probabilities after each
of its decisions (they leak information -- it is an inspection mode).
A human session is far too short for meaningful win rates (a few hundred
hands is roughly +/-100 mbb/h); treat it as a qualitative impression.
'''

import os
import sys
import math
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

import rlcard_patches  # noqa: F401
import rlcard
from agent_loading import load_policy, KINDS, CHECKPOINT_KINDS, CALL, RAISE, FOLD, CHECK
import behaviour_tracker as bt
from hand_browser import HandBrowser

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SESSION_DIR = os.path.join(REPO_ROOT, 'experiments', 'human_sessions')
HUMAN, AGENT = 0, 1          # seats
SUIT_SYMBOL = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
SUIT_COLOUR = {'S': '#111111', 'C': '#111111', 'H': '#c62828', 'D': '#c62828'}
FELT, FELT_EDGE = '#1f6b43', '#14492d'
CARD_W, CARD_H = 62, 88


YOU_VERBS = {'bets': 'bet', 'raises': 'raise', 'calls': 'call', 'checks': 'check', 'folds': 'fold',
             'limps': 'limp'}


def _bb(chips):
    value = chips / bt.BIG_BLIND
    return f'{value:g}'


class PlayGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Limit Hold'em -- play against an agent")
        self.geometry('1220x780')
        self.minsize(1000, 680)

        self.kind_var = tk.StringVar(value='tag')
        self.ckpt_var = tk.StringVar(value='')
        self.seed_var = tk.StringVar(value='')
        self.think_var = tk.BooleanVar(value=False)
        self.reveal_var = tk.BooleanVar(value=True)
        self.delay_var = tk.IntVar(value=700)
        self.status_var = tk.StringVar(value='Choose an opponent and press "New session".')
        self.stats_var = tk.StringVar(value='')

        self.env = rlcard.make('limit-holdem')
        self.policy = None
        self.opponent_name = None
        self.state = None
        self.player = None
        self.session = []
        self.session_file = None
        self.recorder = None
        self.hand_no = 0
        self.hand_over = True
        self.message = ''
        self.last_action = {HUMAN: '', AGENT: ''}
        self.pending_after = None

        self._build_ui()
        self._bind_keys()
        self._redraw()

    # ---------------------------------------------------------------- UI --- #

    def _build_ui(self):
        top = ttk.Frame(self, padding=(8, 8, 8, 4))
        top.pack(side='top', fill='x')
        ttk.Label(top, text='Opponent').pack(side='left')
        kind = ttk.Combobox(top, textvariable=self.kind_var, values=list(KINDS), state='readonly', width=9)
        kind.pack(side='left', padx=(4, 10))
        ttk.Label(top, text='Checkpoint').pack(side='left')
        ttk.Entry(top, textvariable=self.ckpt_var, width=46).pack(side='left', padx=4)
        ttk.Button(top, text='Browse...', command=self._browse_ckpt).pack(side='left')
        ttk.Button(top, text='New session', command=self._new_session).pack(side='left', padx=(10, 0))
        ttk.Separator(top, orient='vertical').pack(side='left', fill='y', padx=10)
        ttk.Label(top, text='Seed').pack(side='left')
        ttk.Entry(top, textvariable=self.seed_var, width=8).pack(side='left', padx=(4, 10))
        ttk.Checkbutton(top, text="Show agent's thinking", variable=self.think_var).pack(side='left')
        ttk.Checkbutton(top, text='Reveal its cards after a fold', variable=self.reveal_var).pack(side='left', padx=8)
        ttk.Label(top, text='Delay (ms)').pack(side='left')
        ttk.Spinbox(top, from_=0, to=3000, increment=100, textvariable=self.delay_var, width=6).pack(side='left', padx=4)

        status = ttk.Frame(self, padding=(8, 0, 8, 6))
        status.pack(side='bottom', fill='x')
        ttk.Label(status, textvariable=self.status_var).pack(side='left')
        bottom = ttk.Frame(self, padding=(8, 4, 8, 4))
        bottom.pack(side='bottom', fill='x')
        self.fold_button = ttk.Button(bottom, text='Fold (F)', width=16, command=lambda: self._human(FOLD))
        self.fold_button.pack(side='left')
        self.call_button = ttk.Button(bottom, text='Check (C)', width=18, command=self._human_call_check)
        self.call_button.pack(side='left', padx=6)
        self.raise_button = ttk.Button(bottom, text='Raise (R)', width=20, command=lambda: self._human(RAISE))
        self.raise_button.pack(side='left')
        self.next_button = ttk.Button(bottom, text='Next hand (N)', width=16, command=self._new_hand)
        self.next_button.pack(side='left', padx=(16, 0))
        ttk.Button(bottom, text='Review / replay...', command=self._open_replay).pack(side='right')
        ttk.Button(bottom, text='Save session as...', command=self._save_as).pack(side='right', padx=6)

        body = ttk.Frame(self, padding=(8, 4, 8, 4))
        body.pack(side='top', fill='both', expand=True)
        self.canvas = tk.Canvas(body, width=760, height=520, background=FELT, highlightthickness=0)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda _e: self._redraw())

        side = ttk.Frame(body, padding=(8, 0, 0, 0), width=380)
        side.pack(side='right', fill='y')
        side.pack_propagate(False)
        stats_box = ttk.LabelFrame(side, text='Session', padding=6)
        stats_box.pack(side='top', fill='x')
        ttk.Label(stats_box, textvariable=self.stats_var, justify='left', font=('Consolas', 9)).pack(anchor='w')
        think_box = ttk.LabelFrame(side, text="Agent's last decision", padding=6)
        think_box.pack(side='top', fill='x', pady=(6, 0))
        self.think_label = ttk.Label(think_box, text='(enable "Show agent\'s thinking")', justify='left',
                                     font=('Consolas', 9), wraplength=350)
        self.think_label.pack(anchor='w')
        hist_box = ttk.LabelFrame(side, text='Hand history', padding=6)
        hist_box.pack(side='top', fill='both', expand=True, pady=(6, 0))
        scroll = ttk.Scrollbar(hist_box, orient='vertical')
        self.history = tk.Text(hist_box, wrap='word', font=('Consolas', 9), yscrollcommand=scroll.set,
                               background='#fafafa')
        scroll.configure(command=self.history.yview)
        scroll.pack(side='right', fill='y')
        self.history.pack(side='left', fill='both', expand=True)
        self.history.tag_configure('hand', foreground='#0d47a1', font=('Consolas', 9, 'bold'))
        self.history.tag_configure('you', foreground='#1b5e20')
        self.history.tag_configure('agent', foreground='#b71c1c')
        self.history.tag_configure('result', font=('Consolas', 9, 'bold'))
        self.history.configure(state='disabled')

        self._set_buttons()

    def _bind_keys(self):
        self.bind('<KeyPress-f>', lambda _e: self._human(FOLD))
        self.bind('<KeyPress-c>', lambda _e: self._human_call_check())
        self.bind('<KeyPress-r>', lambda _e: self._human(RAISE))
        self.bind('<KeyPress-n>', lambda _e: self._new_hand())
        self.bind('<space>', lambda _e: self._new_hand())

    # ------------------------------------------------------------ session --- #

    def _browse_ckpt(self):
        path = filedialog.askopenfilename(title='Select a checkpoint (or cancel to pick a run folder)',
                                          initialdir=os.path.join(REPO_ROOT, 'experiments'),
                                          filetypes=[('Checkpoints', '*.pt *.ckpt'), ('All files', '*.*')])
        if not path:
            path = filedialog.askdirectory(title='Select a run folder',
                                           initialdir=os.path.join(REPO_ROOT, 'experiments'))
        if path:
            self.ckpt_var.set(path)

    def _new_session(self):
        kind = self.kind_var.get()
        ckpt = self.ckpt_var.get().strip() or None
        if kind in CHECKPOINT_KINDS and not ckpt:
            messagebox.showerror('Checkpoint needed', f'A {kind} opponent needs a checkpoint file or run folder.')
            return
        self.status_var.set(f'Loading {kind}...')
        self.update_idletasks()
        try:
            self.policy = load_policy(kind, ckpt if kind in CHECKPOINT_KINDS else None)
        except Exception as exc:
            messagebox.showerror('Could not load the opponent', f'{type(exc).__name__}: {exc}')
            self.status_var.set('Loading failed.')
            return
        self.opponent_name = kind if kind not in CHECKPOINT_KINDS else f'{kind} ({os.path.basename(os.path.normpath(ckpt))})'
        self.session = []
        self.hand_no = 0
        os.makedirs(SESSION_DIR, exist_ok=True)
        self.session_file = os.path.join(SESSION_DIR, f'{kind}_{time.strftime("%Y%m%d_%H%M%S")}.jsonl')
        self.recorder = bt.HandRecorder({'you': 'You', 'agent': self.opponent_name})
        self._clear_history()
        self._update_stats()
        self.status_var.set(f'Playing against {self.opponent_name}. Session log: {self.session_file}')
        self._new_hand()

    def _new_hand(self):
        if self.policy is None or not self.hand_over:
            return
        seed = self.seed_var.get().strip()
        if seed:
            try:
                self.env.seed(int(seed) + self.hand_no)
            except ValueError:
                messagebox.showerror('Seed', 'The seed must be an integer (or empty for random deals).')
                return
        if hasattr(self.policy, 'seed'):
            self.policy.seed(int(time.time() * 1000) % 2 ** 31)
        self.hand_no += 1
        self.state, self.player = self.env.reset()
        self.recorder.start(self.env, {HUMAN: 'you', AGENT: 'agent'}, deal=self.hand_no)
        self.hand_over = False
        self.last_action = {HUMAN: '', AGENT: ''}
        pos = self.recorder.hand['positions']['you']
        self._log(f'\nHand {self.hand_no}: you are {pos}, holding {bt.cards_text(self.recorder.hand["holes"]["you"])}\n',
                  'hand')
        self._advance()

    def _advance(self):
        if self.env.is_over():
            self._finish_hand()
            return
        if self.player == AGENT:
            self.message = f'{self.opponent_name} is thinking...'
            self._set_buttons()
            self._redraw()
            self.pending_after = self.after(max(0, int(self.delay_var.get() or 0)), self._agent_move)
        else:
            raw = self.state['raw_obs']
            to_call = max(raw['all_chips']) - raw['my_chips']
            self.message = f'Your turn -- {_bb(to_call)} BB to call' if to_call else 'Your turn'
            self._set_buttons()
            self._redraw()

    def _agent_move(self):
        self.pending_after = None
        probs = np.asarray(self.policy.probs(self.state), dtype=float)
        action, _ = self.policy.eval_step(self.state)
        record = self.recorder.decision(self.env, self.state, AGENT, action, probs)
        word = bt._action_word(record)
        self.last_action[AGENT] = word
        self._log(f'  {self.opponent_name} {word}\n', 'agent')
        if self.think_var.get():
            p = ', '.join(f'{a} {record["probs"][a]:.2f}' for a in record['legal'])
            self.think_label.configure(text=f'{record["street"]}: {word}\nprobabilities: {p}')
        self.state, self.player = self.env.step(action)
        self._advance()

    def _human_call_check(self):
        if self.state is None:
            return
        legal = self.state['legal_actions']
        self._human(CALL if CALL in legal else CHECK)

    def _human(self, action):
        if self.hand_over or self.player != HUMAN or action not in self.state['legal_actions']:
            return
        record = self.recorder.decision(self.env, self.state, HUMAN, action, None)
        word = YOU_VERBS[bt._action_word(record)]
        self.last_action[HUMAN] = word
        self._log(f'  You {word}\n', 'you')
        self.state, self.player = self.env.step(action)
        self._advance()

    def _finish_hand(self):
        hand = self.recorder.finish(self.env)
        self.session.append(hand)
        self.hand_over = True
        try:
            bt.save_hands(self.session, self.session_file)
        except OSError as exc:
            self.status_var.set(f'Could not auto-save the session: {exc}')
        result = hand['payoff_bb']['you']
        if hand['showdown']:
            bh = hand.get('best_hands', {})
            detail = (f'showdown: you {bh.get("you", "")}, {self.opponent_name} '
                      f'{bt.cards_text(hand["holes"]["agent"])} ({bh.get("agent", "")})')
        else:
            detail = 'you folded' if hand['folded'] == 'you' else f'{self.opponent_name} folded'
        verb = 'win' if result > 0 else ('lose' if result < 0 else 'split')
        self.message = f'You {verb} {abs(result):g} BB -- {detail}.  Press N for the next hand.'
        self._log(f'  result: {result:+g} BB ({detail})\n', 'result')
        self._update_stats()
        self._set_buttons()
        self._redraw()

    # ----------------------------------------------------------- display --- #

    def _set_buttons(self):
        human_turn = (not self.hand_over and self.player == HUMAN and self.state is not None)
        legal = self.state['legal_actions'] if human_turn else []
        raw = self.state['raw_obs'] if self.state is not None else None
        for button, enabled in ((self.fold_button, FOLD in legal and CHECK not in legal),
                                (self.call_button, CALL in legal or CHECK in legal),
                                (self.raise_button, RAISE in legal)):
            button.configure(state='normal' if enabled else 'disabled')
        if human_turn:
            to_call = max(raw['all_chips']) - raw['my_chips']
            self.call_button.configure(text=f'Call {_bb(to_call)} BB (C)' if CALL in legal else 'Check (C)')
            # chips this action adds: the call plus the fixed raise size of the street
            added = to_call + self.env.game.round.raise_amount
            label = f'Raise +{_bb(added)} BB (R)' if to_call else f'Bet {_bb(added)} BB (R)'
            self.raise_button.configure(text=label)
        self.next_button.configure(state='normal' if (self.hand_over and self.policy is not None) else 'disabled')

    def _update_stats(self):
        n = len(self.session)
        if not n:
            self.stats_var.set('no hands yet')
            return
        results = np.array([h['payoff_bb']['you'] for h in self.session])
        mbb = results.mean() * 1000
        se = results.std(ddof=1) * 1000 / math.sqrt(n) if n > 1 else float('nan')
        showdowns = sum(h['showdown'] for h in self.session)
        self.stats_var.set(f'opponent   {self.opponent_name}\n'
                           f'hands      {n}\n'
                           f'total      {results.sum():+g} BB\n'
                           f'win rate   {mbb:+.0f} mbb/h  (+/- {se:.0f})\n'
                           f'showdowns  {showdowns}')

    def _draw_card(self, x, y, card=None, hidden=False, empty=False):
        c = self.canvas
        if empty:
            c.create_rectangle(x, y, x + CARD_W, y + CARD_H, outline='#3f8f62', width=2, dash=(4, 3))
            return
        if hidden or card is None:
            c.create_rectangle(x, y, x + CARD_W, y + CARD_H, fill='#1e3a8a', outline='#e5e7eb', width=2)
            c.create_rectangle(x + 6, y + 6, x + CARD_W - 6, y + CARD_H - 6, outline='#93c5fd', width=1)
            return
        suit, rank = card[0], card[1]
        rank = '10' if rank == 'T' else rank
        colour = SUIT_COLOUR[suit]
        c.create_rectangle(x, y, x + CARD_W, y + CARD_H, fill='white', outline='#9ca3af', width=1)
        c.create_text(x + 8, y + 6, text=rank, anchor='nw', fill=colour, font=('Segoe UI', 14, 'bold'))
        c.create_text(x + CARD_W / 2, y + CARD_H / 2 + 8, text=SUIT_SYMBOL[suit], fill=colour,
                      font=('Segoe UI Symbol', 26))

    def _redraw(self):
        c = self.canvas
        c.delete('all')
        w, h = max(c.winfo_width(), 600), max(c.winfo_height(), 420)
        c.create_oval(30, 30, w - 30, h - 30, fill=FELT, outline=FELT_EDGE, width=14)
        if self.recorder is None or self.recorder.hand is None and not self.session:
            c.create_text(w / 2, h / 2, text='Pick an opponent and press "New session"', fill='white',
                          font=('Segoe UI', 16))
            return
        hand = self.recorder.hand or self.session[-1]
        g = self.env.game
        positions = hand['positions']
        in_chips = {HUMAN: g.players[HUMAN].in_chips, AGENT: g.players[AGENT].in_chips}

        # opponent (top)
        reveal = self.hand_over and (hand['showdown'] or self.reveal_var.get())
        cx = w / 2 - CARD_W - 4
        for i, card in enumerate(hand['holes']['agent']):
            self._draw_card(cx + i * (CARD_W + 8), 60, card, hidden=not reveal)
        c.create_text(w / 2, 40, text=f'{self.opponent_name}  [{positions["agent"]}]', fill='white',
                      font=('Segoe UI', 12, 'bold'))
        c.create_text(w / 2 + CARD_W + 30, 104, text=f'in pot {_bb(in_chips[AGENT])} BB\n{self.last_action[AGENT]}',
                      fill='#fde68a', anchor='w', font=('Segoe UI', 11))

        # board
        board = [card.get_index() for card in g.public_cards] if not self.hand_over else hand['board']
        bx = w / 2 - (5 * CARD_W + 4 * 8) / 2
        for i in range(5):
            if i < len(board):
                self._draw_card(bx + i * (CARD_W + 8), h / 2 - CARD_H / 2 - 10, board[i])
            else:
                self._draw_card(bx + i * (CARD_W + 8), h / 2 - CARD_H / 2 - 10, empty=True)
        pot = in_chips[HUMAN] + in_chips[AGENT]
        c.create_text(w / 2, h / 2 + CARD_H / 2 + 10, text=f'pot {_bb(pot)} BB', fill='white',
                      font=('Segoe UI', 13, 'bold'))

        # you (bottom)
        for i, card in enumerate(hand['holes']['you']):
            self._draw_card(cx + i * (CARD_W + 8), h - 60 - CARD_H, card)
        c.create_text(w / 2, h - 38, text=f'You  [{positions["you"]}]', fill='white', font=('Segoe UI', 12, 'bold'))
        c.create_text(w / 2 + CARD_W + 30, h - 60 - CARD_H / 2, text=f'in pot {_bb(in_chips[HUMAN])} BB\n{self.last_action[HUMAN]}',
                      fill='#fde68a', anchor='w', font=('Segoe UI', 11))
        if self.hand_over and hand['showdown']:
            bh = hand.get('best_hands', {})
            c.create_text(w / 2 - CARD_W - 30, 104, text=bh.get('agent', ''), fill='white', anchor='e',
                          font=('Segoe UI', 11, 'italic'))
            c.create_text(w / 2 - CARD_W - 30, h - 60 - CARD_H / 2, text=bh.get('you', ''), fill='white',
                          anchor='e', font=('Segoe UI', 11, 'italic'))

        c.create_rectangle(40, h / 2 + CARD_H / 2 + 34, w - 40, h / 2 + CARD_H / 2 + 62, fill='#0f3d26', outline='')
        c.create_text(w / 2, h / 2 + CARD_H / 2 + 48, text=self.message, fill='white', font=('Segoe UI', 11))

    def _log(self, text, tag=None):
        self.history.configure(state='normal')
        self.history.insert('end', text, tag or ())
        self.history.configure(state='disabled')
        self.history.see('end')

    def _clear_history(self):
        self.history.configure(state='normal')
        self.history.delete('1.0', 'end')
        self.history.configure(state='disabled')

    # ------------------------------------------------------ save / replay --- #

    def _save_as(self):
        if not self.session:
            messagebox.showinfo('Nothing to save', 'No hands played yet.')
            return
        path = filedialog.asksaveasfilename(title='Save session', defaultextension='.jsonl',
                                            initialdir=SESSION_DIR, filetypes=[('Hand log', '*.jsonl')])
        if path:
            bt.save_hands(self.session, path)
            self.status_var.set(f'Session saved to {path}')

    def _open_replay(self):
        ReplayWindow(self, self.session if self.session else None)


class ReplayWindow(tk.Toplevel):
    ''' Browse the hands of a session or of any behaviour_tracker hands.jsonl. '''

    def __init__(self, parent, hands=None):
        super().__init__(parent)
        self.title('Hand replay')
        self.geometry('1050x680')
        bar = ttk.Frame(self, padding=6)
        bar.pack(side='top', fill='x')
        ttk.Button(bar, text='Open hand log...', command=self._open).pack(side='left')
        self.summary = ttk.Label(bar, text='')
        self.summary.pack(side='left', padx=12)
        self.browser = HandBrowser(self, default_role='you')
        self.browser.pack(fill='both', expand=True, padx=6, pady=(0, 6))
        self.browser.role_box.bind('<<ComboboxSelected>>', lambda _e: (self.browser.refresh(), self._summarise()),
                                   add='+')
        if hands:
            self._set(hands)

    # kept for callers / tests that use the browser's widgets directly
    @property
    def listbox(self):
        return self.browser.listbox

    @property
    def shown(self):
        return self.browser.shown

    @property
    def text(self):
        return self.browser.text

    @property
    def info(self):
        return self.summary

    @property
    def filter_var(self):
        return self.browser.filter_var

    def _fill(self):
        self.browser.refresh()
        self._summarise()

    def _set(self, hands):
        self.browser.set_hands(hands)
        self._summarise()

    def _summarise(self):
        hands, role = self.browser.hands, self.browser.role_var.get()
        if not hands:
            self.summary.configure(text='')
            return
        s = bt.compute_stats(hands, role)
        self.summary.configure(text=f"{len(hands)} hands | {role}: {s['win rate (mbb/h)'][0]:+.0f} mbb/h, "
                                    f"VPIP {s['VPIP'][0]:.0%}, PFR {s['PFR'][0]:.0%}")

    def _open(self):
        path = filedialog.askopenfilename(title='Open hand log', filetypes=[('Hand log', '*.jsonl')],
                                          initialdir=os.path.join(REPO_ROOT, 'experiments'), parent=self)
        if path:
            try:
                self._set(bt.load_hands(path))
            except (OSError, ValueError, KeyError) as exc:
                messagebox.showerror('Cannot read hand log', str(exc), parent=self)


if __name__ == '__main__':
    PlayGUI().mainloop()
