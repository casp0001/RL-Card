''' Behaviour tracker GUI

A Tkinter front-end for behaviour_tracker.py, in the style of train_gui.py and
tune_gui.py:

  Track    pick an agent and an opponent (checkpoints or rule bots), play a
           duplicate match in a subprocess (progress bar, live output), and
           load the result when it finishes
  Stats    both players' style statistics side by side, with the bluff /
           value / slow-play thresholds editable (recomputed instantly)
  Hands    browse, filter and export the recorded hands
  Compare  add any number of hand logs or stats files and compare the agents
           in one table (CSV export, bar chart of selected statistics)

The match itself runs `python behaviour_tracker.py track ...` with the
interpreter in the "Interpreter" field (it needs rlcard and torch); the GUI
only needs rlcard (for the hand-strength code) and matplotlib for charts.

Run with:  python tracker_gui.py
'''

import os
import sys
import csv
import json
import time
import queue
import codecs
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import behaviour_tracker as bt
from hand_browser import HandBrowser
from train_gui import Tooltip, _quote, _fmt_elapsed

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TRACKER = os.path.join(REPO_ROOT, 'behaviour_tracker.py')
OUT_ROOT = os.path.join(REPO_ROOT, 'experiments', 'behaviour')
KINDS = ('dqn', 'nfsp', 'deepcfr', 'ppo', 'tag', 'nit', 'lag', 'honest', 'station', 'maniac', 'random')
CHECKPOINT_KINDS = ('dqn', 'nfsp', 'deepcfr', 'ppo')
THRESHOLD_HELP = {
    'bluff_equity': 'A post-flop bet / raise with equity below this (and no draw) counts as a bluff.',
    'value_equity': 'Aggression with equity at or above this counts as a value bet.',
    'slowplay_equity': 'Checking / calling with equity at or above this (raising allowed) is a slow-play.',
    'semi_hs': 'Semi-bluff: current hand strength below this ...',
    'semi_potential': '... and equity minus current strength at least this (drawing equity).',
}


def _short_path(path):
    ''' Path relative to the repo when inside it, else absolute. '''
    path = os.path.abspath(path)
    return os.path.relpath(path, REPO_ROOT) if path.startswith(REPO_ROOT + os.sep) else path


def _fmt_stat(name, value):
    if value is None:
        return 'n/a'
    if 'mbb' in name:
        return f'{value:+.1f}'
    if 'factor' in name:
        return f'{value:.2f}'
    return f'{value:.1%}'


class TrackerGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('RLCard Behaviour Tracker')
        self.geometry('1180x820')
        self.minsize(900, 620)

        self.interp_var = tk.StringVar(value=sys.executable)
        self.kind_var = tk.StringVar(value='tag')
        self.ckpt_var = tk.StringVar()
        self.opp_var = tk.StringVar(value='lag')
        self.opp_ckpt_var = tk.StringVar()
        self.deals_var = tk.StringVar(value='500')
        self.out_var = tk.StringVar()
        self.status_var = tk.StringVar(value='Idle')
        self.loaded_var = tk.StringVar(value='No hand log loaded.')
        self.threshold_vars = {k: tk.StringVar(value=str(v)) for k, v in bt.THRESHOLDS.items()}

        self.proc = None
        self.queue = queue.Queue()
        self.start_time = None
        self.hands = []
        self.hands_path = None
        self.compare_entries = []    # (label, hands or None, stats dict)

        self._build_ui()
        self._suggest_out()
        self.after(100, self._drain)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    # ---------------------------------------------------------------- UI --- #

    def _build_ui(self):
        top = ttk.Frame(self, padding=(8, 8, 8, 4))
        top.pack(side='top', fill='x')
        ttk.Label(top, text='Interpreter').pack(side='left')
        entry = ttk.Entry(top, textvariable=self.interp_var, width=70)
        entry.pack(side='left', padx=6)
        Tooltip(entry, 'Python used to run the match; it needs rlcard and torch.')
        ttk.Button(top, text='Browse...', command=self._browse_interpreter).pack(side='left')
        ttk.Label(top, textvariable=self.loaded_var, foreground='#555').pack(side='right')

        status = ttk.Frame(self, padding=(8, 2, 8, 6))
        status.pack(side='bottom', fill='x')
        ttk.Label(status, textvariable=self.status_var).pack(side='left')

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=8, pady=4)
        self.notebook.add(self._build_track_tab(), text='Track')
        self.notebook.add(self._build_stats_tab(), text='Stats')
        self.browser = HandBrowser(self.notebook, default_role='agent')
        self.notebook.add(self.browser, text='Hands')
        self.notebook.add(self._build_compare_tab(), text='Compare')

    def _player_row(self, parent, row, label, kind_var, ckpt_var):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=4)
        box = ttk.Combobox(parent, textvariable=kind_var, values=KINDS, state='readonly', width=10)
        box.grid(row=row, column=1, sticky='w', padx=6)
        box.bind('<<ComboboxSelected>>', lambda _e: self._suggest_out())
        ttk.Label(parent, text='checkpoint').grid(row=row, column=2, sticky='e', padx=(10, 4))
        ttk.Entry(parent, textvariable=ckpt_var, width=60).grid(row=row, column=3, sticky='we')
        ttk.Button(parent, text='File...', command=lambda: self._browse_ckpt(ckpt_var, False)).grid(
            row=row, column=4, padx=(6, 0))
        ttk.Button(parent, text='Folder...', command=lambda: self._browse_ckpt(ckpt_var, True)).grid(
            row=row, column=5, padx=(4, 0))

    def _build_track_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        form = ttk.LabelFrame(frame, text='Match', padding=8)
        form.pack(side='top', fill='x')
        self._player_row(form, 0, 'Agent', self.kind_var, self.ckpt_var)
        self._player_row(form, 1, 'Opponent', self.opp_var, self.opp_ckpt_var)
        ttk.Label(form, text='Duplicate deals').grid(row=2, column=0, sticky='w', pady=4)
        deals = ttk.Entry(form, textvariable=self.deals_var, width=10)
        deals.grid(row=2, column=1, sticky='w', padx=6)
        Tooltip(deals, 'Each deal is played twice with the seats swapped. ~500 deals is a quick look, '
                       '2000+ gives stable statistics.')
        ttk.Label(form, text='Output folder').grid(row=3, column=0, sticky='w', pady=4)
        ttk.Entry(form, textvariable=self.out_var, width=80).grid(row=3, column=1, columnspan=3, sticky='we', padx=6)
        ttk.Button(form, text='Browse...', command=self._browse_out).grid(row=3, column=4, padx=(6, 0))
        form.columnconfigure(3, weight=1)

        buttons = ttk.Frame(frame, padding=(0, 8, 0, 4))
        buttons.pack(side='top', fill='x')
        self.run_button = ttk.Button(buttons, text='Run match', command=self._run)
        self.run_button.pack(side='left')
        self.stop_button = ttk.Button(buttons, text='Stop', command=self._stop, state='disabled')
        self.stop_button.pack(side='left', padx=6)
        ttk.Button(buttons, text='Open hand log...', command=self._open_log).pack(side='left', padx=(16, 0))
        ttk.Button(buttons, text='Open output folder', command=self._open_folder).pack(side='right')
        self.progress = ttk.Progressbar(buttons, length=260, mode='determinate')
        self.progress.pack(side='left', padx=16)

        cmd_frame = ttk.LabelFrame(frame, text='Command', padding=6)
        cmd_frame.pack(side='top', fill='x')
        self.cmd_text = tk.Text(cmd_frame, height=2, wrap='word', font=('Consolas', 9))
        self.cmd_text.pack(fill='x')
        for var in (self.kind_var, self.ckpt_var, self.opp_var, self.opp_ckpt_var, self.deals_var,
                    self.out_var, self.interp_var):
            var.trace_add('write', lambda *_a: self._update_command())

        log_frame = ttk.LabelFrame(frame, text='Output', padding=6)
        log_frame.pack(side='top', fill='both', expand=True, pady=(6, 0))
        scroll = ttk.Scrollbar(log_frame, orient='vertical')
        self.log = tk.Text(log_frame, wrap='none', font=('Consolas', 9), background='#1e1e1e',
                           foreground='#d4d4d4', yscrollcommand=scroll.set, height=14)
        scroll.configure(command=self.log.yview)
        scroll.pack(side='right', fill='y')
        self.log.pack(side='left', fill='both', expand=True)
        self.log.tag_configure('meta', foreground='#4ec9b0')
        self.log.tag_configure('error', foreground='#f48771')
        self.log.configure(state='disabled')
        return frame

    def _build_stats_tab(self):
        frame = ttk.Frame(self.notebook, padding=8)
        bar = ttk.LabelFrame(frame, text='Thresholds (hand classes for the bluff / value statistics)', padding=6)
        bar.pack(side='top', fill='x')
        for i, (key, var) in enumerate(self.threshold_vars.items()):
            lab = ttk.Label(bar, text=key)
            lab.grid(row=0, column=2 * i, sticky='e', padx=(8, 4))
            ent = ttk.Entry(bar, textvariable=var, width=6)
            ent.grid(row=0, column=2 * i + 1, sticky='w')
            Tooltip(lab, THRESHOLD_HELP[key])
            Tooltip(ent, THRESHOLD_HELP[key])
        ttk.Button(bar, text='Apply', command=self._apply_thresholds).grid(row=0, column=2 * len(self.threshold_vars),
                                                                         padx=10)
        ttk.Button(bar, text='Defaults', command=self._default_thresholds).grid(
            row=0, column=2 * len(self.threshold_vars) + 1)
        ttk.Button(bar, text='Add to Compare', command=self._add_current_to_compare).grid(
            row=0, column=2 * len(self.threshold_vars) + 2, padx=(20, 0))

        columns = ('stat', 'agent', 'agent_n', 'opponent', 'opponent_n')
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill='both', expand=True, pady=(8, 0))
        self.stats_tree = ttk.Treeview(tree_frame, columns=columns, show='headings')
        for col, title, width, anchor in (('stat', 'statistic', 330, 'w'), ('agent', 'agent', 120, 'e'),
                                          ('agent_n', 'n', 70, 'e'), ('opponent', 'opponent', 120, 'e'),
                                          ('opponent_n', 'n', 70, 'e')):
            self.stats_tree.heading(col, text=title)
            self.stats_tree.column(col, width=width, anchor=anchor, stretch=(col == 'stat'))
        ys = ttk.Scrollbar(tree_frame, orient='vertical', command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=ys.set)
        ys.pack(side='right', fill='y')
        self.stats_tree.pack(side='left', fill='both', expand=True)
        self.stats_tree.tag_configure('section', background='#eef2f7')
        ttk.Label(frame, text='n = hands or decisions behind each rate (probability-weighted rates use the '
                              'policy\'s action probabilities). Small n -> treat the number with caution.',
                  foreground='#555').pack(side='top', anchor='w', pady=(6, 0))
        return frame

    def _build_compare_tab(self):
        frame = ttk.Frame(self.notebook, padding=8)
        bar = ttk.Frame(frame)
        bar.pack(side='top', fill='x')
        ttk.Button(bar, text='Add hand log / stats file...', command=self._add_compare_files).pack(side='left')
        ttk.Button(bar, text='Add current (agent)', command=lambda: self._add_current_to_compare('agent')).pack(
            side='left', padx=6)
        ttk.Button(bar, text='Add current (opponent)',
                   command=lambda: self._add_current_to_compare('opponent')).pack(side='left')
        ttk.Button(bar, text='Remove column', command=self._remove_compare).pack(side='left', padx=(16, 0))
        ttk.Button(bar, text='Clear', command=self._clear_compare).pack(side='left', padx=6)
        ttk.Button(bar, text='Export CSV...', command=self._export_compare).pack(side='right')
        ttk.Button(bar, text='Chart selected rows', command=self._chart_compare).pack(side='right', padx=6)
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill='both', expand=True, pady=(8, 0))
        self.compare_tree = ttk.Treeview(tree_frame, show='headings', selectmode='extended')
        ys = ttk.Scrollbar(tree_frame, orient='vertical', command=self.compare_tree.yview)
        xs = ttk.Scrollbar(tree_frame, orient='horizontal', command=self.compare_tree.xview)
        self.compare_tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side='right', fill='y')
        xs.pack(side='bottom', fill='x')
        self.compare_tree.pack(side='left', fill='both', expand=True)
        ttk.Label(frame, text='Columns computed from hand logs follow the current thresholds; stats.json files '
                              'keep the thresholds they were computed with. Select rows (Ctrl/Shift-click) to chart them.',
                  foreground='#555').pack(side='top', anchor='w', pady=(6, 0))
        self._fill_compare()
        return frame

    # ------------------------------------------------------------ track --- #

    def _suggest_out(self):
        self.out_var.set(os.path.join('experiments', 'behaviour',
                                      f'{self.kind_var.get()}_vs_{self.opp_var.get()}_{time.strftime("%Y%m%d_%H%M")}'))

    def command(self):
        kind, opp = self.kind_var.get(), self.opp_var.get()
        if kind in CHECKPOINT_KINDS and not self.ckpt_var.get().strip():
            raise ValueError(f'the agent ({kind}) needs a checkpoint')
        if opp in CHECKPOINT_KINDS and not self.opp_ckpt_var.get().strip():
            raise ValueError(f'the opponent ({opp}) needs a checkpoint')
        deals = int(self.deals_var.get())
        if deals < 2:
            raise ValueError('use at least 2 deals')
        argv = [self.interp_var.get().strip() or sys.executable, '-u', TRACKER, 'track',
                '--kind', kind, '--opponent', opp, '--deals', str(deals),
                '--out', self.out_var.get().strip() or OUT_ROOT, '--show', '0']
        if kind in CHECKPOINT_KINDS:
            argv += ['--ckpt', self.ckpt_var.get().strip()]
        if opp in CHECKPOINT_KINDS:
            argv += ['--opp_ckpt', self.opp_ckpt_var.get().strip()]
        return argv

    def _update_command(self):
        try:
            text = ' '.join(_quote(a) for a in self.command())
        except ValueError as exc:
            text = f'! {exc}'
        self.cmd_text.configure(state='normal')
        self.cmd_text.delete('1.0', 'end')
        self.cmd_text.insert('1.0', text)
        self.cmd_text.configure(state='disabled')

    def _run(self):
        if self.proc is not None:
            return
        try:
            argv = self.command()
        except ValueError as exc:
            messagebox.showerror('Invalid settings', str(exc))
            return
        self._append('$ ' + ' '.join(_quote(a) for a in argv), 'meta')
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        try:
            self.proc = subprocess.Popen(argv, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, bufsize=0, env=env,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except OSError as exc:
            messagebox.showerror('Could not start', str(exc))
            return
        self.run_out = os.path.join(REPO_ROOT, self.out_var.get().strip())
        self.start_time = time.time()
        self.progress.configure(value=0, maximum=100)
        self.run_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()
        self._tick()

    def _read(self, proc):
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        buf = ''
        while True:
            try:
                chunk = os.read(proc.stdout.fileno(), 8192)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            buf += decoder.decode(chunk).replace('\r\n', '\n').replace('\r', '\n')
            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                self.queue.put(('line', line))
        if buf:
            self.queue.put(('line', buf))
        self.queue.put(('exit', proc.wait()))

    def _drain(self):
        try:
            for _ in range(2000):
                kind, payload = self.queue.get_nowait()
                if kind == 'line':
                    self._on_line(payload)
                else:
                    self._on_exit(payload)
        except queue.Empty:
            pass
        self.after(100, self._drain)

    def _on_line(self, line):
        if 'INFO - Restoring' in line or not line.strip():
            return
        stripped = line.strip()
        if stripped.endswith(' deals') and '/' in stripped:
            try:
                done, total = stripped.split()[0].split('/')
                self.progress.configure(value=100 * int(done) / int(total))
            except ValueError:
                pass
        self._append(line, 'error' if ('Traceback' in line or 'Error' in line) else None)

    def _on_exit(self, code):
        elapsed = _fmt_elapsed(time.time() - (self.start_time or time.time()))
        self.proc = None
        self.run_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        if code == 0:
            self.progress.configure(value=100)
            self._append(f'--- finished in {elapsed} ---', 'meta')
            self.status_var.set(f'Match finished in {elapsed}')
            path = os.path.join(self.run_out, 'hands.jsonl')
            if os.path.isfile(path):
                self.load_log(path)
                self.notebook.select(1)
        else:
            self._append(f'--- exited with code {code} after {elapsed} ---', 'error')
            self.status_var.set(f'Match failed (exit {code}) -- see Output')

    def _tick(self):
        if self.proc is None:
            return
        self.status_var.set(f'Running match -- {_fmt_elapsed(time.time() - self.start_time)}')
        self.after(1000, self._tick)

    def _stop(self):
        if self.proc is not None:
            self._append('--- stop requested ---', 'meta')
            try:
                self.proc.terminate()
            except OSError:
                pass

    def _append(self, text, tag=None):
        self.log.configure(state='normal')
        self.log.insert('end', text + '\n', tag or ())
        self.log.configure(state='disabled')
        self.log.see('end')

    # ------------------------------------------------------------- stats --- #

    def load_log(self, path):
        try:
            hands = bt.load_hands(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror('Cannot read hand log', str(exc))
            return
        if not hands:
            messagebox.showerror('Empty hand log', path)
            return
        self.hands, self.hands_path = hands, path
        names = hands[0]['names']
        roles = sorted(hands[0]['holes'])
        self.roles = roles
        self.loaded_var.set(f'{len(hands)} hands: ' + ' vs '.join(f'{names[r]} ({r})' for r in roles)
                            + f'  --  {_short_path(path)}')
        self.browser.default_role = roles[0] if 'agent' not in roles else 'agent'
        self.browser.set_hands(hands)
        self._fill_stats()

    def _thresholds_from_form(self):
        values = {}
        for key, var in self.threshold_vars.items():
            try:
                values[key] = float(var.get())
            except ValueError:
                raise ValueError(f'{key} must be a number')
        return values

    def _apply_thresholds(self):
        try:
            values = self._thresholds_from_form()
        except ValueError as exc:
            messagebox.showerror('Invalid threshold', str(exc))
            return
        # behaviour_tracker reads THRESHOLDS as its default everywhere (stats,
        # hand tags, filters), so it is updated in place
        bt.THRESHOLDS.update(values)
        self._fill_stats()
        self.browser.refresh()
        self._recompute_compare()
        self.status_var.set('Thresholds applied: ' + ', '.join(f'{k}={v:g}' for k, v in values.items()))

    def _default_thresholds(self):
        defaults = {'bluff_equity': 0.35, 'value_equity': 0.6, 'slowplay_equity': 0.8,
                    'semi_hs': 0.5, 'semi_potential': 0.08}
        for key, value in defaults.items():
            self.threshold_vars[key].set(str(value))
        self._apply_thresholds()

    def _fill_stats(self):
        self.stats_tree.delete(*self.stats_tree.get_children())
        if not self.hands:
            return
        roles = self.roles
        stats = {r: bt.compute_stats(self.hands, r) for r in roles}
        names = self.hands[0]['names']
        first, second = roles[0], roles[-1]
        if 'agent' in roles:
            first, second = 'agent', next(r for r in roles if r != 'agent')
        self.stats_roles = (first, second)
        self.stats_tree.heading('agent', text=f'{names[first]} ({first})')
        self.stats_tree.heading('opponent', text=f'{names[second]} ({second})')
        section_starts = {'VPIP': 'pre-flop', 'flop: bet when checked to': 'post-flop',
                          'WTSD': 'showdown', 'bluff rate (bet/raise | weak hand)': 'bluffing'}
        for name in stats[first]:
            if name in section_starts:
                self.stats_tree.insert('', 'end', values=(f'-- {section_starts[name]} --', '', '', '', ''),
                                       tags=('section',))
            a, an = stats[first][name]
            b, bn = stats[second][name]
            self.stats_tree.insert('', 'end', values=(name, _fmt_stat(name, a), f'{an:.0f}',
                                                      _fmt_stat(name, b), f'{bn:.0f}'))

    # ----------------------------------------------------------- compare --- #

    def _add_current_to_compare(self, role=None):
        if not self.hands:
            messagebox.showinfo('Nothing loaded', 'Run a match or open a hand log first.')
            return
        role = role or getattr(self, 'stats_roles', ('agent',))[0]
        if role not in self.roles:
            role = self.roles[0]
        label = self.hands[0]['names'][role]
        self._add_compare(f'{label} vs {self._other_name(role)}', self.hands, role)
        self.notebook.select(3)

    def _other_name(self, role):
        names = self.hands[0]['names']
        return next(names[r] for r in self.roles if r != role)

    def _add_compare_files(self):
        paths = filedialog.askopenfilenames(title='Add hand logs or stats files',
                                            initialdir=OUT_ROOT if os.path.isdir(OUT_ROOT) else REPO_ROOT,
                                            filetypes=[('Hand logs / stats', '*.jsonl *.json'), ('All', '*.*')])
        for path in paths:
            try:
                if path.endswith('.jsonl'):
                    hands = bt.load_hands(path)
                    roles = sorted(hands[0]['holes'])
                    role = 'agent' if 'agent' in roles else roles[0]
                    names = hands[0]['names']
                    other = next(names[r] for r in roles if r != role)
                    self._add_compare(f'{names[role]} vs {other}', hands, role)
                else:
                    with open(path, encoding='utf-8') as f:
                        data = json.load(f)
                    stats = {k: (v['value'], v['n']) for k, v in data['stats'].items()}
                    self.compare_entries.append((data.get('name', os.path.basename(path)), None, None, stats))
            except (OSError, ValueError, KeyError, IndexError) as exc:
                messagebox.showerror('Cannot add', f'{path}\n{exc}')
        self._fill_compare()

    def _add_compare(self, label, hands, role):
        existing = {e[0] for e in self.compare_entries}
        base, k = label, 2
        while label in existing:
            label, k = f'{base} ({k})', k + 1
        self.compare_entries.append((label, hands, role, bt.compute_stats(hands, role)))
        self._fill_compare()

    def _recompute_compare(self):
        self.compare_entries = [(l, h, r, bt.compute_stats(h, r) if h is not None else s)
                                for l, h, r, s in self.compare_entries]
        self._fill_compare()

    def _remove_compare(self):
        if self.compare_entries:
            self.compare_entries.pop()
            self._fill_compare()

    def _clear_compare(self):
        self.compare_entries = []
        self._fill_compare()

    def _fill_compare(self):
        tree = self.compare_tree
        tree.delete(*tree.get_children())
        cols = ['stat'] + [f'c{i}' for i in range(len(self.compare_entries))]
        tree.configure(columns=cols)
        tree.heading('stat', text='statistic')
        tree.column('stat', width=320, anchor='w', stretch=False)
        for i, (label, *_rest) in enumerate(self.compare_entries):
            tree.heading(f'c{i}', text=label)
            tree.column(f'c{i}', width=max(110, 8 * len(label)), anchor='e', stretch=False)
        if not self.compare_entries:
            return
        for name in self.compare_entries[0][3]:
            cells = [_fmt_stat(name, e[3].get(name, (None, 0))[0]) for e in self.compare_entries]
            tree.insert('', 'end', iid=name, values=[name] + cells)

    def _export_compare(self):
        if not self.compare_entries:
            return
        path = filedialog.asksaveasfilename(title='Export comparison', defaultextension='.csv',
                                            filetypes=[('CSV', '*.csv')])
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['statistic'] + [e[0] for e in self.compare_entries]
                       + [f'{e[0]} (n)' for e in self.compare_entries])
            for name in self.compare_entries[0][3]:
                values = [e[3].get(name, (None, 0)) for e in self.compare_entries]
                w.writerow([name] + ['' if v is None else v for v, _ in values] + [n for _, n in values])
        self.status_var.set(f'Comparison exported to {path}')

    def _chart_compare(self):
        rows = list(self.compare_tree.selection())
        if not rows or not self.compare_entries:
            messagebox.showinfo('Nothing selected', 'Select one or more statistics (rows) to chart.')
            return
        import matplotlib
        matplotlib.use('TkAgg')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        window = tk.Toplevel(self)
        window.title('Comparison chart')
        fig = Figure(figsize=(9, 4.8), dpi=100)
        ax = fig.add_subplot(111)
        n_entries = len(self.compare_entries)
        width = 0.8 / n_entries
        percent = all('mbb' not in r and 'factor' not in r for r in rows)
        for i, (label, *_rest, stats) in enumerate(self.compare_entries):
            values = [stats.get(r, (None, 0))[0] for r in rows]
            values = [float('nan') if v is None else (100 * v if percent else v) for v in values]
            ax.bar([x + i * width - 0.4 + width / 2 for x in range(len(rows))], values, width, label=label)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(rows, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('%' if percent else 'value')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        def save():
            path = filedialog.asksaveasfilename(parent=window, defaultextension='.png',
                                                filetypes=[('PNG', '*.png'), ('PDF', '*.pdf')])
            if path:
                fig.savefig(path, dpi=150)
        ttk.Button(window, text='Save figure...', command=save).pack(side='bottom', pady=4)
        self._last_chart = (window, fig)

    # -------------------------------------------------------------- misc --- #

    def _browse_ckpt(self, var, folder):
        start = os.path.join(REPO_ROOT, 'experiments')
        path = (filedialog.askdirectory(title='Select a run folder', initialdir=start) if folder else
                filedialog.askopenfilename(title='Select a checkpoint', initialdir=start,
                                           filetypes=[('Checkpoints', '*.pt *.ckpt'), ('All files', '*.*')]))
        if path:
            var.set(path)

    def _browse_out(self):
        path = filedialog.askdirectory(title='Output folder', initialdir=REPO_ROOT)
        if path:
            self.out_var.set(path)

    def _browse_interpreter(self):
        path = filedialog.askopenfilename(title='Select Python interpreter',
                                          filetypes=[('Python executable', 'python*.exe'), ('All files', '*.*')]
                                          if os.name == 'nt' else [('All files', '*.*')])
        if path:
            self.interp_var.set(path)

    def _open_log(self):
        path = filedialog.askopenfilename(title='Open hand log', filetypes=[('Hand log', '*.jsonl')],
                                          initialdir=OUT_ROOT if os.path.isdir(OUT_ROOT) else REPO_ROOT)
        if path:
            self.load_log(path)
            self.notebook.select(1)

    def _open_folder(self):
        target = os.path.join(REPO_ROOT, self.out_var.get().strip())
        if not os.path.isdir(target):
            target = OUT_ROOT if os.path.isdir(OUT_ROOT) else REPO_ROOT
        try:
            os.startfile(target)
        except (AttributeError, OSError):
            subprocess.Popen(['xdg-open', target])

    def _on_close(self):
        if self.proc is not None:
            if not messagebox.askyesno('Match running', 'A match is still running. Stop it and quit?'):
                return
            try:
                self.proc.kill()
            except OSError:
                pass
        self.destroy()


if __name__ == '__main__':
    TrackerGUI().mainloop()
