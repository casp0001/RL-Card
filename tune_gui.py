''' Hyper-parameter tuning GUI

A Tkinter front-end for tune.py, in the style of train_gui.py. It edits the
study settings and the search space, starts several tune.py worker
processes on one study (they share an Optuna journal file, so they work on
the same study in parallel), shows their output, a live trial table, and runs
the `finalists` and `report` steps.

Like train_gui.py it uses only the standard library and talks to tune.py
through subprocesses, so it runs from any interpreter -- pick the one that
has rlcard, torch and optuna in the "Interpreter" field.

Run with:  python tune_gui.py
'''

import os
import sys
import json
import queue
import ctypes
import codecs
import threading
import subprocess
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import tune_spaces
from train_gui import Tooltip, ScrollableFrame, _quote, _fmt_elapsed

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TUNE_ROOT = os.path.join(REPO_ROOT, 'experiments', 'tune')
TUNE_SCRIPT = os.path.join(REPO_ROOT, 'tune.py')
MAX_LOG_LINES = 5000
REFRESH_MS = 30_000

STATE_COLOURS = {'COMPLETE': '#1b5e20', 'PRUNED': '#8d6e00', 'FAIL': '#b71c1c', 'RUNNING': '#0d47a1'}

SETTINGS = [
    # key, label, kind, help
    ('trials', 'Total trials', 'int', 'Trials of the whole study (all workers), incl. pruned and failed.'),
    ('workers', 'Parallel workers', 'int', 'tune.py processes started on this study. Each runs one trial '
                                           'at a time; they share the study through its journal file.'),
    ('threads', 'Threads / worker', 'int', 'torch / OpenMP threads per worker. 0 = cores / workers.'),
    ('budget', 'Budget / trial', 'int', 'Episodes per trial (Deep CFR: iterations).'),
    ('evals', 'Evaluations / trial', 'int', 'Panel evaluations per trial; each is a pruning checkpoint.'),
    ('eval_deals', 'Deals / evaluation', 'int', 'Duplicate deals per panel member per evaluation '
                                                '(500 -> about +/-65 mbb/h vs the TAG).'),
    ('eval_panel', 'Panel members', 'str', 'Comma-separated eval_panel members.'),
    ('seed', 'Training seed', 'int', 'Seed of every trial (finalists use new seeds).'),
    ('prune', 'Pruning', 'bool', 'Successive halving: stop trials that fall behind at a checkpoint.'),
    ('min_resource_frac', 'Min. fraction before pruning', 'float',
     'A trial cannot be pruned before this fraction of its evaluations (RL curves cross early).'),
    ('reduction_factor', 'Reduction factor', 'int', 'Keep about 1/rf of trials at each pruning rung.'),
    ('startup_trials', 'Random startup trials', 'int', 'Random trials before the TPE model is used.'),
]
# settings that are stored in the study's config.json (locked for an existing study)
CONFIG_SETTINGS = ('budget', 'evals', 'eval_deals', 'eval_panel', 'seed', 'prune',
                   'min_resource_frac', 'reduction_factor', 'startup_trials')


def available_ram_gb():
    ''' (total, available) physical memory in GB, or (None, None). '''
    if os.name == 'nt':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in ('total', 'avail', 'tp', 'ap', 'tv', 'av', 'ae')]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.total / 1e9, status.avail / 1e9
        return None, None
    try:
        pages, size = os.sysconf('SC_PHYS_PAGES'), os.sysconf('SC_PAGE_SIZE')
        avail = os.sysconf('SC_AVPHYS_PAGES')
        return pages * size / 1e9, avail * size / 1e9
    except (ValueError, OSError, AttributeError):
        return None, None


class TuneGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('RLCard Hyper-parameter Tuning')
        self.geometry('1180x860')
        self.minsize(900, 640)

        self.workers = {}          # id -> Popen
        self.side_proc = None      # finalists / report process
        self.queue = queue.Queue()
        self.start_time = None
        self.space_rows = {}       # param -> dict of tk variables
        self.locked = False

        self.algo_var = tk.StringVar(value='dqn')
        self.study_var = tk.StringVar(value='')
        self.interp_var = tk.StringVar(value=sys.executable)
        self.status_var = tk.StringVar(value='Idle')
        self.mem_var = tk.StringVar(value='')
        self.summary_var = tk.StringVar(value='')
        self.autorefresh_var = tk.BooleanVar(value=True)
        self.autoscroll_var = tk.BooleanVar(value=True)
        self.setting_vars = {key: tk.StringVar() for key, *_ in SETTINGS}
        self.top_var = tk.StringVar(value='3')
        self.seeds_var = tk.StringVar(value='3')
        self.final_deals_var = tk.StringVar(value='5000')

        self._build_ui()
        self._on_algo_change(reset_study=True)
        self.after(100, self._drain_output)
        self.after(REFRESH_MS, self._auto_refresh)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    # ---------------------------------------------------------------- UI --- #

    def _build_ui(self):
        top = ttk.Frame(self, padding=(8, 8, 8, 4))
        top.pack(side='top', fill='x')

        ttk.Label(top, text='Algorithm').grid(row=0, column=0, sticky='w', padx=(0, 6))
        algo_box = ttk.Combobox(top, textvariable=self.algo_var, state='readonly', width=12,
                                values=sorted(tune_spaces.ALGORITHMS))
        algo_box.grid(row=0, column=1, sticky='w')
        algo_box.bind('<<ComboboxSelected>>', lambda _e: self._on_algo_change(reset_study=True))

        ttk.Label(top, text='Study').grid(row=0, column=2, sticky='w', padx=(16, 6))
        study_entry = ttk.Entry(top, textvariable=self.study_var, width=30)
        study_entry.grid(row=0, column=3, sticky='w')
        study_entry.bind('<Return>', lambda _e: self._load_study())
        study_entry.bind('<FocusOut>', lambda _e: self._load_study())
        Tooltip(study_entry, 'Name of the study (folder experiments/tune/<study>). An existing study '
                             'is loaded with its settings locked; Start then adds trials to it.')
        ttk.Button(top, text='Load / new', command=self._load_study).grid(row=0, column=4, padx=4)
        ttk.Button(top, text='Browse studies...', command=self._browse_study).grid(row=0, column=5)
        self.lock_label = ttk.Label(top, text='', foreground='#8d6e00')
        self.lock_label.grid(row=0, column=6, sticky='w', padx=10)

        ttk.Label(top, text='Interpreter').grid(row=1, column=0, sticky='w', pady=(6, 0))
        interp_entry = ttk.Entry(top, textvariable=self.interp_var, width=70)
        interp_entry.grid(row=1, column=1, columnspan=5, sticky='we', pady=(6, 0))
        Tooltip(interp_entry, 'Python used for tune.py; it needs rlcard, torch, optuna and matplotlib.')
        ib = ttk.Frame(top)
        ib.grid(row=1, column=6, sticky='w', padx=6, pady=(6, 0))
        ttk.Button(ib, text='Browse...', command=self._browse_interpreter).pack(side='left')
        ttk.Button(ib, text='Check', command=self._check_interpreter).pack(side='left', padx=(4, 0))
        top.columnconfigure(6, weight=1)

        panes = ttk.PanedWindow(self, orient='vertical')
        panes.pack(side='top', fill='both', expand=True, padx=8, pady=4)

        notebook = ttk.Notebook(panes)
        panes.add(notebook, weight=3)
        notebook.add(self._build_settings_tab(notebook), text='Study settings')
        notebook.add(self._build_space_tab(notebook), text='Search space')
        notebook.add(self._build_trials_tab(notebook), text='Trials')
        self.notebook = notebook

        mid = ttk.Frame(panes)
        cmd_frame = ttk.LabelFrame(mid, text='Worker command', padding=6)
        cmd_frame.pack(side='top', fill='x')
        self.cmd_text = tk.Text(cmd_frame, height=2, wrap='word', font=('Consolas', 9))
        self.cmd_text.pack(fill='x')
        self.cmd_text.configure(state='disabled')

        buttons = ttk.Frame(mid, padding=(0, 6, 0, 0))
        buttons.pack(side='top', fill='x')
        self.start_button = ttk.Button(buttons, text='Start study', command=self._start)
        self.start_button.pack(side='left')
        self.stop_button = ttk.Button(buttons, text='Stop workers', command=self._stop, state='disabled')
        self.stop_button.pack(side='left', padx=(6, 0))
        ttk.Separator(buttons, orient='vertical').pack(side='left', fill='y', padx=10)
        ttk.Label(buttons, text='Finalists: top').pack(side='left')
        ttk.Entry(buttons, textvariable=self.top_var, width=4).pack(side='left', padx=(4, 6))
        ttk.Label(buttons, text='seeds').pack(side='left')
        ttk.Entry(buttons, textvariable=self.seeds_var, width=4).pack(side='left', padx=(4, 6))
        ttk.Label(buttons, text='deals').pack(side='left')
        ttk.Entry(buttons, textvariable=self.final_deals_var, width=7).pack(side='left', padx=(4, 6))
        self.finalists_button = ttk.Button(buttons, text='Run finalists', command=self._run_finalists)
        self.finalists_button.pack(side='left')
        Tooltip(self.finalists_button, 'Re-train the best completed trials with new seeds at the full '
                                       'budget and evaluate them with many deals. Pick the winner here.')
        self.report_button = ttk.Button(buttons, text='Report + plots', command=self._run_report)
        self.report_button.pack(side='left', padx=(6, 0))
        ttk.Button(buttons, text='Open study folder', command=self._open_study).pack(side='right')
        panes.add(mid, weight=0)

        log_frame = ttk.LabelFrame(panes, text='Output', padding=6)
        bar = ttk.Frame(log_frame)
        bar.pack(side='top', fill='x', pady=(0, 4))
        ttk.Checkbutton(bar, text='Auto-scroll', variable=self.autoscroll_var).pack(side='left')
        ttk.Button(bar, text='Clear', command=self._clear_log).pack(side='left', padx=(10, 0))
        ttk.Button(bar, text='Save log...', command=self._save_log).pack(side='left', padx=(6, 0))
        inner = ttk.Frame(log_frame)
        inner.pack(fill='both', expand=True)
        scroll = ttk.Scrollbar(inner, orient='vertical')
        self.log = tk.Text(inner, wrap='none', font=('Consolas', 9), background='#1e1e1e',
                           foreground='#d4d4d4', yscrollcommand=scroll.set, height=12)
        scroll.configure(command=self.log.yview)
        scroll.pack(side='right', fill='y')
        self.log.pack(side='left', fill='both', expand=True)
        self.log.tag_configure('meta', foreground='#4ec9b0')
        self.log.tag_configure('error', foreground='#f48771')
        self.log.tag_configure('trial', foreground='#dcdcaa')
        self.log.configure(state='disabled')
        panes.add(log_frame, weight=2)

        status = ttk.Frame(self, padding=(8, 2, 8, 6))
        status.pack(side='bottom', fill='x')
        ttk.Label(status, textvariable=self.status_var).pack(side='left')
        ttk.Label(status, textvariable=self.mem_var, foreground='#555').pack(side='right')

    def _build_settings_tab(self, parent):
        frame = ttk.Frame(parent, padding=8)
        self.setting_widgets = {}
        for index, (key, label, kind, help_text) in enumerate(SETTINGS):
            row, col = index // 2, (index % 2) * 2
            lab = ttk.Label(frame, text=label)
            lab.grid(row=row, column=col, sticky='w', padx=(4, 6), pady=4)
            var = self.setting_vars[key]
            if kind == 'bool':
                widget = ttk.Checkbutton(frame, variable=var, onvalue='1', offvalue='0')
            else:
                widget = ttk.Entry(frame, textvariable=var, width=36 if key == 'eval_panel' else 14)
            widget.grid(row=row, column=col + 1, sticky='w', padx=(0, 24), pady=4)
            var.trace_add('write', lambda *_a: self._update_derived())
            Tooltip(lab, help_text)
            Tooltip(widget, help_text)
            self.setting_widgets[key] = widget
        note = ('Each trial trains one configuration and reports the panel "min" after every evaluation; '
                'a trial\'s score is the mean of its last 3 reports. Every worker uses the same seed, '
                'the panel uses common random numbers, so trials face identical evaluation deals.')
        ttk.Label(frame, text=note, foreground='#555', wraplength=900, justify='left').grid(
            row=len(SETTINGS) // 2 + 1, column=0, columnspan=4, sticky='w', padx=4, pady=(12, 0))
        return frame

    def _build_space_tab(self, parent):
        outer = ttk.Frame(parent, padding=6)
        bar = ttk.Frame(outer)
        bar.pack(side='top', fill='x', pady=(0, 4))
        ttk.Button(bar, text='Reset to defaults', command=self._reset_space).pack(side='left')
        ttk.Button(bar, text='Save space...', command=self._save_space).pack(side='left', padx=(6, 0))
        ttk.Button(bar, text='Load space...', command=self._load_space).pack(side='left', padx=(6, 0))
        ttk.Label(bar, text='Untick "Tune" to fix a parameter at its value. Choices: comma-separated.',
                  foreground='#555').pack(side='left', padx=12)
        self.space_frame = ScrollableFrame(outer)
        self.space_frame.pack(fill='both', expand=True)
        return outer

    def _build_trials_tab(self, parent):
        frame = ttk.Frame(parent, padding=6)
        bar = ttk.Frame(frame)
        bar.pack(side='top', fill='x', pady=(0, 4))
        ttk.Button(bar, text='Refresh', command=self._refresh_trials).pack(side='left')
        ttk.Checkbutton(bar, text='Auto-refresh (30 s) while running',
                        variable=self.autorefresh_var).pack(side='left', padx=10)
        ttk.Label(bar, textvariable=self.summary_var).pack(side='left', padx=10)
        columns = ('number', 'state', 'score', 'evals', 'last', 'time', 'params')
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill='both', expand=True)
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=10)
        widths = {'number': 50, 'state': 80, 'score': 80, 'evals': 50, 'last': 80, 'time': 80, 'params': 700}
        titles = {'number': '#', 'state': 'state', 'score': 'score', 'evals': 'evals',
                  'last': 'last min', 'time': 'time', 'params': 'parameters'}
        for c in columns:
            self.tree.heading(c, text=titles[c], command=lambda c=c: self._sort_tree(c))
            self.tree.column(c, width=widths[c], anchor='w' if c == 'params' else 'e',
                             stretch=(c == 'params'))
        for state, colour in STATE_COLOURS.items():
            self.tree.tag_configure(state, foreground=colour)
        ys = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=ys.set)
        ys.pack(side='right', fill='y')
        self.tree.pack(side='left', fill='both', expand=True)
        self.tree.bind('<Double-1>', self._open_trial)
        Tooltip(self.tree, 'Double-click a trial to open its log folder (panel.csv / panel.png).')
        self.finalist_text = tk.Text(frame, height=6, wrap='none', font=('Consolas', 9))
        self.finalist_text.pack(side='top', fill='x', pady=(6, 0))
        self.finalist_text.configure(state='disabled')
        self._sort_key = ('score', True)
        return frame

    # --------------------------------------------------- algo / study --- #

    def _on_algo_change(self, reset_study=False):
        algo = self.algo_var.get()
        if reset_study:
            self.study_var.set(f'{algo}_{time.strftime("%Y%m%d_%H%M")}')
        defaults = tune_spaces.DEFAULT_SETTINGS[algo]
        values = {'trials': '40', 'workers': '2', 'threads': '0', 'budget': str(defaults['budget']),
                  'evals': str(defaults['evals']), 'eval_deals': '500',
                  'eval_panel': 'random,station,maniac,nit,tag,lag', 'seed': '42', 'prune': '1',
                  'min_resource_frac': '0.25', 'reduction_factor': '3', 'startup_trials': '12'}
        for key, value in values.items():
            self.setting_vars[key].set(value)
        self._set_locked(False)
        self._populate_space(tune_spaces.default_space(algo))
        self._load_study(quiet=True)

    def _study_dir(self):
        return os.path.join(TUNE_ROOT, self.study_var.get().strip())

    def _load_study(self, quiet=False):
        name = self.study_var.get().strip()
        path = os.path.join(TUNE_ROOT, name, 'config.json')
        if not name or not os.path.isfile(path):
            if self.locked:
                self._set_locked(False)
            self._update_derived()
            return
        try:
            with open(path, encoding='utf-8') as f:
                cfg = json.load(f)
        except (OSError, ValueError) as exc:
            if not quiet:
                messagebox.showerror('Cannot read study', str(exc))
            return
        if cfg.get('algo') != self.algo_var.get():
            self.algo_var.set(cfg['algo'])
        for key in CONFIG_SETTINGS:
            value = cfg.get(key)
            if key == 'prune':
                value = '1' if value else '0'
            self.setting_vars[key].set(str(value))
        self._populate_space(cfg['space'])
        self._set_locked(True, cfg.get('created', ''))
        self._refresh_trials()

    def _set_locked(self, locked, created=''):
        self.locked = locked
        state = 'disabled' if locked else 'normal'
        for key in CONFIG_SETTINGS:
            self.setting_widgets[key].configure(state=state)
        for row in self.space_rows.values():
            for widget in row['widgets']:
                widget.configure(state=state)
        self.lock_label.configure(text=(f'existing study (created {created}) -- settings locked; '
                                        f'Start adds trials' if locked else 'new study'))
        self._update_derived()

    def _browse_study(self):
        os.makedirs(TUNE_ROOT, exist_ok=True)
        path = filedialog.askdirectory(title='Select a study folder', initialdir=TUNE_ROOT)
        if path:
            self.study_var.set(os.path.basename(os.path.normpath(path)))
            self._load_study()

    # ------------------------------------------------------ search space --- #

    def _populate_space(self, space):
        for child in self.space_frame.inner.winfo_children():
            child.destroy()
        self.space_rows = {}
        grid = self.space_frame.inner
        headers = ['Tune', 'Parameter', 'Type', 'Low', 'High', 'Log', 'Choices', 'Fixed value', 'Targets']
        for col, text in enumerate(headers):
            ttk.Label(grid, text=text, font=('Segoe UI', 9, 'bold')).grid(row=0, column=col, sticky='w',
                                                                         padx=4, pady=(0, 4))
        for r, (name, spec) in enumerate(space.items(), start=1):
            row = {'spec': spec}
            row['tune'] = tk.StringVar(value='1' if spec.get('tune', True) else '0')
            row['low'] = tk.StringVar(value=_fmt_num(spec.get('low', '')))
            row['high'] = tk.StringVar(value=_fmt_num(spec.get('high', '')))
            row['log'] = tk.StringVar(value='1' if spec.get('log') else '0')
            row['choices'] = tk.StringVar(value=', '.join(str(c) for c in spec.get('choices', [])))
            default_value = spec.get('value', spec.get('start', (spec.get('choices') or [spec.get('low', '')])[0]))
            row['value'] = tk.StringVar(value=str(default_value))
            widgets = [ttk.Checkbutton(grid, variable=row['tune'], onvalue='1', offvalue='0')]
            widgets[0].grid(row=r, column=0, padx=4)
            name_label = ttk.Label(grid, text=name)
            name_label.grid(row=r, column=1, sticky='w', padx=4)
            if spec.get('help'):
                Tooltip(name_label, spec['help'])
            ttk.Label(grid, text=spec['type'] + (' (layers)' if spec.get('list') else '')).grid(
                row=r, column=2, sticky='w', padx=4)
            numeric = spec['type'] in ('float', 'int')
            for col, key, width in ((3, 'low', 11), (4, 'high', 11)):
                entry = ttk.Entry(grid, textvariable=row[key], width=width)
                entry.grid(row=r, column=col, padx=4, pady=2)
                if not numeric:
                    entry.configure(state='disabled')
                else:
                    widgets.append(entry)
            log_box = ttk.Checkbutton(grid, variable=row['log'], onvalue='1', offvalue='0')
            log_box.grid(row=r, column=5, padx=4)
            if numeric:
                widgets.append(log_box)
            else:
                log_box.configure(state='disabled')
            choices = ttk.Entry(grid, textvariable=row['choices'], width=34)
            choices.grid(row=r, column=6, padx=4, pady=2)
            if numeric:
                choices.configure(state='disabled')
            else:
                widgets.append(choices)
            value_entry = ttk.Entry(grid, textvariable=row['value'], width=14)
            value_entry.grid(row=r, column=7, padx=4, pady=2)
            widgets.append(value_entry)
            ttk.Label(grid, text=', '.join(spec.get('targets', [name])), foreground='#555').grid(
                row=r, column=8, sticky='w', padx=4)
            row['widgets'] = widgets
            for key in ('tune', 'low', 'high', 'log', 'choices', 'value'):
                row[key].trace_add('write', lambda *_a: self._update_derived())
            self.space_rows[name] = row
        self._update_derived()

    def current_space(self):
        ''' The search space as edited in the form. Raises ValueError. '''
        space = {}
        for name, row in self.space_rows.items():
            spec = dict(row['spec'])
            kind = spec['type']
            spec['tune'] = row['tune'].get() == '1'
            if kind in ('float', 'int'):
                cast = float if kind == 'float' else int
                try:
                    spec['low'] = cast(float(row['low'].get()))
                    spec['high'] = cast(float(row['high'].get()))
                except ValueError:
                    raise ValueError(f'{name}: low / high must be numbers')
                spec['log'] = row['log'].get() == '1'
            else:
                original = row['spec'].get('choices', [])
                cast = type(original[0]) if original and not spec.get('list') else str
                try:
                    spec['choices'] = [cast(c.strip()) for c in row['choices'].get().split(',') if c.strip()]
                except ValueError:
                    raise ValueError(f'{name}: choices must be {cast.__name__}s')
            if not spec['tune']:
                raw = row['value'].get().strip()
                try:
                    if kind == 'float':
                        spec['value'] = float(raw)
                    elif kind == 'int':
                        spec['value'] = int(float(raw))
                    else:
                        original = row['spec'].get('choices', [])
                        cast = type(original[0]) if original and not spec.get('list') else str
                        spec['value'] = cast(raw)
                except ValueError:
                    raise ValueError(f'{name}: fixed value {raw!r} has the wrong type')
            else:
                spec.pop('value', None)
            space[name] = spec
        tune_spaces.validate_space(space)
        return space

    def _reset_space(self):
        if self.locked:
            return
        self._populate_space(tune_spaces.default_space(self.algo_var.get()))

    def _save_space(self):
        try:
            space = self.current_space()
        except ValueError as exc:
            messagebox.showerror('Invalid search space', str(exc))
            return
        path = filedialog.asksaveasfilename(title='Save search space', defaultextension='.json',
                                            initialdir=TUNE_ROOT, filetypes=[('JSON', '*.json')],
                                            initialfile=f'{self.algo_var.get()}_space.json')
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'algo': self.algo_var.get(), 'space': space}, f, indent=2)
            self.status_var.set('Search space saved to ' + path)

    def _load_space(self):
        if self.locked:
            messagebox.showinfo('Study locked', 'An existing study keeps its search space.')
            return
        path = filedialog.askopenfilename(title='Load search space', initialdir=TUNE_ROOT,
                                          filetypes=[('JSON', '*.json')])
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            space = data.get('space', data)
            tune_spaces.validate_space(space)
        except (OSError, ValueError) as exc:
            messagebox.showerror('Cannot load search space', str(exc))
            return
        if data.get('algo') and data['algo'] != self.algo_var.get():
            self.algo_var.set(data['algo'])
            self._on_algo_change(reset_study=True)
        self._populate_space(space)

    # ---------------------------------------------------------- derived --- #

    def _int_setting(self, key):
        return int(float(self.setting_vars[key].get()))

    def _threads(self):
        threads = self._int_setting('threads')
        if threads <= 0:
            threads = max(1, (os.cpu_count() or 2) // max(1, self._int_setting('workers')))
        return threads

    def worker_command(self, fail_stale=False):
        ''' argv of one worker; raises ValueError for invalid fields. '''
        study = self.study_var.get().strip()
        if not study or any(c in study for c in '\\/:*?"<>|'):
            raise ValueError('enter a study name without path characters')
        for key, _label, kind, _help in SETTINGS:
            raw = self.setting_vars[key].get().strip()
            if kind == 'int':
                int(float(raw))
            elif kind == 'float':
                float(raw)
        self.current_space()
        interpreter = self.interp_var.get().strip() or sys.executable
        v = {k: var.get().strip() for k, var in self.setting_vars.items()}
        argv = [interpreter, '-u', TUNE_SCRIPT, 'run', '--algo', self.algo_var.get(), '--study', study,
                '--trials', str(int(float(v['trials']))), '--budget', str(int(float(v['budget']))),
                '--evals', str(int(float(v['evals']))), '--eval_deals', str(int(float(v['eval_deals']))),
                '--eval_panel', v['eval_panel'], '--seed', str(int(float(v['seed']))),
                '--space', os.path.join(TUNE_ROOT, study, 'space.json'),
                '--prune', '1' if v['prune'] == '1' else '0',
                '--min_resource_frac', v['min_resource_frac'],
                '--reduction_factor', str(int(float(v['reduction_factor']))),
                '--startup_trials', str(int(float(v['startup_trials']))),
                '--threads', str(self._threads())]
        if fail_stale:
            argv.append('--fail_stale')
        return argv

    def _update_derived(self):
        try:
            argv = self.worker_command()
            text = ' '.join(_quote(a) for a in argv)
        except (ValueError, KeyError) as exc:
            text = f'! {exc}'
        self.cmd_text.configure(state='normal')
        self.cmd_text.delete('1.0', 'end')
        self.cmd_text.insert('1.0', text)
        self.cmd_text.configure(state='disabled')
        try:
            per_worker = tune_spaces.estimate_memory_gb(self.algo_var.get(), self.current_space())
            workers = self._int_setting('workers')
            total, avail = available_ram_gb()
            text = (f'buffers <= {per_worker:.1f} GB per worker, {per_worker * workers:.1f} GB '
                    f'for {workers} worker(s) (+~1.5 GB each for Python/torch)')
            if total:
                text += f'  |  RAM {avail:.1f} of {total:.1f} GB free'
            self.mem_var.set(text)
        except (ValueError, KeyError):
            self.mem_var.set('')

    # --------------------------------------------------------- processes --- #

    def _env(self, threads):
        env = dict(os.environ)
        env.update({'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8',
                    'OMP_NUM_THREADS': str(threads), 'MKL_NUM_THREADS': str(threads)})
        return env

    def _spawn(self, argv, tag, threads):
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        proc = subprocess.Popen(argv, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, bufsize=0, env=self._env(threads),
                                creationflags=creationflags)
        threading.Thread(target=self._read_output, args=(proc, tag), daemon=True).start()
        return proc

    def _start(self):
        if self.workers:
            messagebox.showinfo('Already running', 'Workers are already running.')
            return
        try:
            space = self.current_space()
            argv = self.worker_command(fail_stale=True)
        except ValueError as exc:
            messagebox.showerror('Invalid settings', str(exc))
            return
        if not os.path.isfile(argv[0]):
            messagebox.showerror('Interpreter not found', argv[0])
            return
        workers = max(1, self._int_setting('workers'))
        per_worker = tune_spaces.estimate_memory_gb(self.algo_var.get(), space)
        _total, avail = available_ram_gb()
        if avail and per_worker * workers + 1.5 * workers > avail and not messagebox.askyesno(
                'Memory', f'The buffers alone may need up to {per_worker * workers:.1f} GB '
                          f'(+ ~{1.5 * workers:.0f} GB for the processes), but only {avail:.1f} GB '
                          f'is free. Start anyway?'):
            return

        os.makedirs(self._study_dir(), exist_ok=True)
        with open(os.path.join(self._study_dir(), 'space.json'), 'w', encoding='utf-8') as f:
            json.dump(space, f, indent=2)

        threads = self._threads()
        self.start_time = time.time()
        self._append(f'--- starting {workers} worker(s) on study {self.study_var.get()} '
                     f'({threads} thread(s) each) ---', 'meta')
        self._append('$ ' + ' '.join(_quote(a) for a in argv), 'meta')
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')

        def launch(index):
            # only the first worker cleans up trials left RUNNING by killed workers
            cmd = self.worker_command(fail_stale=(index == 1))
            try:
                self.workers[index] = self._spawn(cmd, f'w{index}', threads)
            except OSError as exc:
                self._append(f'[w{index}] could not start: {exc}', 'error')
            if index < workers:
                # stagger starts so the first worker creates the study first
                self.after(3000, lambda: launch(index + 1))
        launch(1)
        self._tick_status()

    def _stop(self):
        if not self.workers:
            return
        if not messagebox.askyesno('Stop workers', 'Stop all workers? Their current trials are '
                                                   'lost (they are marked FAIL on the next start); '
                                                   'finished trials stay in the study.'):
            return
        for index, proc in list(self.workers.items()):
            self._append(f'[w{index}] stop requested', 'meta')
            try:
                proc.terminate()
            except OSError:
                pass

    def _run_side(self, argv, label):
        if self.side_proc is not None:
            messagebox.showinfo('Busy', 'A finalists / report process is already running.')
            return
        if not os.path.isfile(os.path.join(self._study_dir(), 'config.json')):
            messagebox.showerror('No study', 'This study has not been started yet.')
            return
        self._append(f'--- {label} ---', 'meta')
        self._append('$ ' + ' '.join(_quote(a) for a in argv), 'meta')
        threads = os.cpu_count() or 2
        self.side_proc = self._spawn(argv, label, threads)
        self.finalists_button.configure(state='disabled')
        self.report_button.configure(state='disabled')

    def _run_finalists(self):
        try:
            top, seeds, deals = (int(self.top_var.get()), int(self.seeds_var.get()),
                                 int(self.final_deals_var.get()))
        except ValueError:
            messagebox.showerror('Invalid values', 'top / seeds / deals must be integers')
            return
        if self.workers and not messagebox.askyesno(
                'Workers running', 'Finalists use the trials completed so far and compete with the '
                                   'workers for CPU. Run anyway?'):
            return
        interpreter = self.interp_var.get().strip() or sys.executable
        self._run_side([interpreter, '-u', TUNE_SCRIPT, 'finalists', '--study', self.study_var.get().strip(),
                        '--top', str(top), '--seeds', str(seeds), '--final_deals', str(deals),
                        '--threads', str(os.cpu_count() or 2)], 'finalists')

    def _run_report(self):
        interpreter = self.interp_var.get().strip() or sys.executable
        self._run_side([interpreter, '-u', TUNE_SCRIPT, 'report', '--study', self.study_var.get().strip()],
                       'report')

    def _read_output(self, proc, tag):
        ''' Stream a child's output line by line; a bare '\\r' (progress
        line) discards the partial line instead of piling it up. '''
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        fd = proc.stdout.fileno()
        current = ''
        pending_cr = False
        while True:
            try:
                chunk = os.read(fd, 8192)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            for char in decoder.decode(chunk):
                if pending_cr:
                    pending_cr = False
                    if char == '\n':
                        self.queue.put(('line', tag, current))
                        current = ''
                        continue
                    current = ''
                if char == '\r':
                    pending_cr = True
                elif char == '\n':
                    self.queue.put(('line', tag, current))
                    current = ''
                elif char != '\x00':
                    current += char
        if current:
            self.queue.put(('line', tag, current))
        self.queue.put(('exit', tag, proc.wait()))

    def _drain_output(self):
        lines = []
        exits = []
        statuses = []
        try:
            for _ in range(2000):
                kind, tag, payload = self.queue.get_nowait()
                if kind == 'line':
                    lines.append((tag, payload))
                elif kind == 'status':
                    statuses.append((tag, payload))
                else:
                    exits.append((tag, payload))
        except queue.Empty:
            pass
        for tag, line in lines:
            if not line.strip():
                continue
            style = None
            if '=== trial' in line or '=== finalist' in line:
                style = 'trial'
            elif 'Traceback' in line or 'Error' in line or 'failed' in line:
                style = 'error'
            self._append(f'[{tag}] {line}', style)
            if '=== trial' in line and ('finished' in line or 'pruned' in line):
                self._refresh_trials()
        for tag, code in exits:
            self._on_process_exit(tag, code)
        for study, data in statuses:
            self._show_trials(study, data)
        self.after(100, self._drain_output)

    def _on_process_exit(self, tag, code):
        style = 'meta' if code == 0 else 'error'
        self._append(f'[{tag}] exited with code {code}', style)
        if tag.startswith('w'):
            self.workers.pop(int(tag[1:]), None)
            if not self.workers:
                elapsed = _fmt_elapsed(time.time() - (self.start_time or time.time()))
                self._append(f'--- all workers finished ({elapsed}) ---', 'meta')
                self.status_var.set(f'Workers finished after {elapsed}')
                self.start_button.configure(state='normal')
                self.stop_button.configure(state='disabled')
                self._load_study(quiet=True)
        else:
            self.side_proc = None
            self.finalists_button.configure(state='normal')
            self.report_button.configure(state='normal')
            if tag == 'report' and code == 0:
                plots = os.path.join(self._study_dir(), 'plots')
                if os.path.isdir(plots) and messagebox.askyesno('Report', 'Open the plots folder?'):
                    _open_path(plots)
        self._refresh_trials()

    def _tick_status(self):
        if not self.workers:
            return
        elapsed = _fmt_elapsed(time.time() - (self.start_time or time.time()))
        self.status_var.set(f'{len(self.workers)} worker(s) running on {self.study_var.get()} -- {elapsed}')
        self.after(1000, self._tick_status)

    # ------------------------------------------------------------ trials --- #

    def _auto_refresh(self):
        if self.autorefresh_var.get() and (self.workers or self.side_proc):
            self._refresh_trials()
        self.after(REFRESH_MS, self._auto_refresh)

    def _refresh_trials(self):
        study = self.study_var.get().strip()
        if not study or not os.path.isfile(os.path.join(TUNE_ROOT, study, 'journal.log')):
            return
        interpreter = self.interp_var.get().strip() or sys.executable

        def work():
            try:
                result = subprocess.run([interpreter, TUNE_SCRIPT, 'status', '--study', study, '--json'],
                                        capture_output=True, text=True, timeout=120, cwd=REPO_ROOT,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                data = json.loads(result.stdout.strip().splitlines()[-1])
            except Exception:
                return
            # Tk may only be touched from the main thread: hand over via the queue
            self.queue.put(('status', study, data))
        threading.Thread(target=work, daemon=True).start()

    def _show_trials(self, study, data):
        if study != self.study_var.get().strip():
            return
        self._trials = data['trials']
        self._fill_tree()
        counts = {}
        for t in data['trials']:
            counts[t['state']] = counts.get(t['state'], 0) + 1
        complete = [t for t in data['trials'] if t['state'] == 'COMPLETE']
        best = max(complete, key=lambda t: t['value'], default=None)
        text = ', '.join(f'{k.lower()} {v}' for k, v in sorted(counts.items()))
        if best:
            text += f'   |   best: trial {best["number"]} ({best["value"]:+.1f})'
        self.summary_var.set(text)

        lines = []
        fin = data.get('finalists')
        if fin and fin.get('runs'):
            by_trial, failed = {}, {}
            for r in fin['runs']:
                by_trial.setdefault(r['trial'], [])
                if 'failed' in r:
                    failed[r['trial']] = failed.get(r['trial'], 0) + 1
                else:
                    by_trial[r['trial']].append(r['final_min'])
            lines.append(f'Finalists ({fin.get("final_deals")} deals per member):')
            for trial, mins in sorted(by_trial.items(),
                                      key=lambda kv: -sum(kv[1]) / len(kv[1]) if kv[1] else float('inf')):
                note = f'  ({failed[trial]} seed(s) failed)' if trial in failed else ''
                if not mins:
                    lines.append(f'  trial {trial:3d}: no successful seed{note}')
                    continue
                mean = sum(mins) / len(mins)
                lines.append(f'  trial {trial:3d}: final panel min {mean:+8.1f} '
                             f'over {len(mins)} seed(s): {", ".join(f"{m:+.0f}" for m in mins)}{note}')
            if fin.get('winner'):
                lines.append(f'  winner: trial {fin["winner"]["trial"]}  {json.dumps(fin["winner"]["params"])}')
        self.finalist_text.configure(state='normal')
        self.finalist_text.delete('1.0', 'end')
        self.finalist_text.insert('1.0', '\n'.join(lines) or 'No finalist runs yet.')
        self.finalist_text.configure(state='disabled')

    def _fill_tree(self):
        trials = list(getattr(self, '_trials', []))
        key, reverse = self._sort_key
        column_value = {
            'number': lambda t: t['number'], 'state': lambda t: t['state'],
            'score': lambda t: t['value'] if t['value'] is not None else float('-inf'),
            'evals': lambda t: t['evals'],
            'last': lambda t: t['last_min'] if t['last_min'] is not None else float('-inf'),
            'time': lambda t: t['seconds'] or 0, 'params': lambda t: json.dumps(t['params'])}
        trials.sort(key=column_value[key], reverse=reverse)
        self.tree.delete(*self.tree.get_children())
        for t in trials:
            params = '  '.join(f'{k}={_fmt_num(v)}' for k, v in t['params'].items())
            self.tree.insert('', 'end', iid=str(t['number']), tags=(t['state'],), values=(
                t['number'], t['state'].lower(),
                '' if t['value'] is None else f'{t["value"]:+.1f}', t['evals'],
                '' if t['last_min'] is None else f'{t["last_min"]:+.0f}',
                _fmt_elapsed(t['seconds']) if t['seconds'] else '', params))
        self._log_dirs = {str(t['number']): t.get('log_dir') for t in trials}

    def _sort_tree(self, column):
        key, reverse = self._sort_key
        self._sort_key = (column, not reverse if key == column else column in ('score', 'last'))
        self._fill_tree()

    def _open_trial(self, _event):
        selected = self.tree.focus()
        log_dir = getattr(self, '_log_dirs', {}).get(selected)
        if log_dir:
            _open_path(os.path.join(REPO_ROOT, log_dir))

    # -------------------------------------------------------------- misc --- #

    def _append(self, text, tag=None):
        self.log.configure(state='normal')
        self.log.insert('end', text + '\n', tag or ())
        total = int(self.log.index('end-1c').split('.')[0])
        if total > MAX_LOG_LINES:
            self.log.delete('1.0', f'{total - MAX_LOG_LINES + 1}.0')
        self.log.configure(state='disabled')
        if self.autoscroll_var.get():
            self.log.see('end')

    def _clear_log(self):
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.configure(state='disabled')

    def _save_log(self):
        path = filedialog.asksaveasfilename(title='Save log', defaultextension='.txt',
                                            filetypes=[('Text', '*.txt')])
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.log.get('1.0', 'end-1c'))

    def _open_study(self):
        target = self._study_dir() if os.path.isdir(self._study_dir()) else TUNE_ROOT
        os.makedirs(target, exist_ok=True)
        _open_path(target)

    def _browse_interpreter(self):
        path = filedialog.askopenfilename(title='Select Python interpreter',
                                          filetypes=[('Python executable', 'python*.exe'), ('All files', '*.*')]
                                          if os.name == 'nt' else [('All files', '*.*')])
        if path:
            self.interp_var.set(path)
            self._update_derived()

    def _check_interpreter(self):
        interpreter = self.interp_var.get().strip() or sys.executable
        probe = ('import sys; print("python", sys.version.split()[0])\n'
                 'for m in ("rlcard", "torch", "numpy", "matplotlib", "optuna", "sklearn"):\n'
                 '    try:\n'
                 '        mod = __import__(m); print("  OK  ", m, getattr(mod, "__version__", ""))\n'
                 '    except Exception as exc:\n'
                 '        print("  MISS", m, "->", exc)\n')
        try:
            result = subprocess.run([interpreter, '-c', probe], capture_output=True, text=True,
                                    timeout=120, cwd=REPO_ROOT)
        except (OSError, subprocess.TimeoutExpired) as exc:
            messagebox.showerror('Check failed', str(exc))
            return
        report = (result.stdout or '') + (result.stderr or '')
        self._append(f'--- interpreter check: {interpreter} ---', 'meta')
        for line in report.splitlines():
            # scikit-learn is optional (parameter importances)
            self._append(line, 'error' if 'MISS' in line and 'sklearn' not in line else 'meta')
        self.status_var.set('Interpreter OK' if 'MISS' not in report.replace('MISS sklearn', '')
                            else 'Interpreter is missing packages -- see Output')

    def _on_close(self):
        running = list(self.workers.values()) + ([self.side_proc] if self.side_proc else [])
        if running:
            if not messagebox.askyesno('Tuning in progress',
                                       'Workers are still running. Stop them and quit? Their current '
                                       'trials are lost; finished trials stay in the study.'):
                return
            for proc in running:
                try:
                    proc.kill()
                except OSError:
                    pass
        self.destroy()


def _fmt_num(value):
    if isinstance(value, float):
        return f'{value:.4g}'
    return str(value)


def _open_path(path):
    try:
        if os.name == 'nt':
            os.startfile(path)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])
    except OSError as exc:
        messagebox.showerror('Could not open folder', str(exc))


if __name__ == '__main__':
    TuneGUI().mainloop()
