''' Training launcher GUI

A small Tkinter front-end for the RLCard training scripts in this repository
(limit_ppo_selfplay.py, limit_dqn_selfplay_custom.py, the NFSP scripts, ...).

It does not import or modify the training scripts. Instead it reads each
script's argparse definition statically with the `ast` module, builds a form
from it, and then launches the script as an ordinary subprocess:

    <interpreter> -u <script.py> --flag value --flag value ...

Because the form is generated from the source, adding or renaming an
add_argument() in a training script shows up here automatically.

Only the standard library is used, so this runs even in an interpreter that
does not have rlcard or torch installed -- pick the interpreter that does have
them in the "Interpreter" field (there is a "Check" button next to it).

Run with:  python train_gui.py
'''

import ast
import codecs
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PRESET_DIR = os.path.join(REPO_ROOT, 'gui_presets')

# Keep at most this many committed lines in the log pane. The NFSP script
# prints one progress line per episode straight to stdout, which over a
# 2,000,000 episode run is far more than a Text widget can hold.
MAX_LOG_LINES = 4000

# Upper bound on committed lines rendered in a single 80 ms refresh, so a
# script that floods stdout cannot lock up the UI thread.
MAX_LINES_PER_TICK = 400

_UNKNOWN = object()


# --------------------------------------------------------------------------- #
# Reading argparse definitions out of a script without importing it
# --------------------------------------------------------------------------- #

class ArgSpec(object):
    ''' One command-line option discovered in a training script. '''

    def __init__(self, flag, type_name=None, default=None, nargs=None,
                 choices=None, help_text='', action=None, raw_default=None):
        self.flag = flag                  # e.g. '--learning_rate'
        self.type_name = type_name        # 'int' | 'float' | 'str' | 'bool' | None
        self.default = default            # evaluated default, or _UNKNOWN
        self.nargs = nargs
        self.choices = choices
        self.help_text = help_text
        self.action = action
        self.raw_default = raw_default    # source text, for defaults we cannot evaluate

    @property
    def name(self):
        return self.flag.lstrip('-')

    @property
    def is_flag(self):
        ''' A store_true / store_false switch, which takes no value. '''
        return self.action in ('store_true', 'store_false')

    @property
    def is_bool(self):
        ''' type=bool / type=str2bool (functions.py) -- see value_to_argv(). '''
        return self.type_name in ('bool', 'str2bool')

    @property
    def is_list(self):
        if isinstance(self.nargs, str) and self.nargs in ('+', '*'):
            return True
        if isinstance(self.nargs, int) and self.nargs > 1:
            return True
        return isinstance(self.default, (list, tuple))

    def default_as_text(self):
        ''' The default rendered for an entry widget. '''
        if self.default is _UNKNOWN or self.default is None:
            return ''
        if isinstance(self.default, (list, tuple)):
            return ' '.join(str(x) for x in self.default)
        return str(self.default)


def _safe_eval(node):
    ''' Evaluate a default-value AST node without executing the script.

    Handles literals plus the int()/float()/str()/bool() wrappers that appear
    in these scripts (e.g. default=int(6e6)). Anything else returns _UNKNOWN,
    and the field is left blank so the script's own default applies.
    '''
    try:
        return ast.literal_eval(node)
    except Exception:
        pass
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        builtins = {'int': int, 'float': float, 'str': str, 'bool': bool}
        if node.func.id in builtins and len(node.args) == 1:
            try:
                return builtins[node.func.id](ast.literal_eval(node.args[0]))
            except Exception:
                pass
    return _UNKNOWN


def _node_source(node):
    try:
        return ast.unparse(node)
    except Exception:
        return '<unknown>'


def parse_script(path):
    ''' Extract the parser description and option list from a script's source.

    Args:
        path (str): path to a training script

    Returns:
        (tuple) (description (str), specs (list of ArgSpec))
    '''
    with open(path, 'r', encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=path)

    description = ''
    specs = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # argparse.ArgumentParser("...") / ArgumentParser(description="...")
        func = node.func
        is_parser_ctor = (
            (isinstance(func, ast.Attribute) and func.attr == 'ArgumentParser')
            or (isinstance(func, ast.Name) and func.id == 'ArgumentParser')
        )
        if is_parser_ctor and not description:
            if node.args:
                value = _safe_eval(node.args[0])
                if isinstance(value, str):
                    description = value
            for kw in node.keywords:
                if kw.arg in ('description', 'prog'):
                    value = _safe_eval(kw.value)
                    if isinstance(value, str) and not description:
                        description = value
            continue

        # Shared helpers that add options from another module, e.g.
        # eval_panel.add_panel_args(parser): read that function's options too
        if isinstance(func, ast.Name) and func.id in HELPER_ARG_FUNCTIONS:
            specs.extend(_helper_specs(*HELPER_ARG_FUNCTIONS[func.id]))
            continue

        if not (isinstance(func, ast.Attribute) and func.attr == 'add_argument'):
            continue

        spec = _spec_from_call(node)
        if spec is not None:
            specs.append(spec)

    return description, specs


# function name -> (module file, function name) of helpers that add options
HELPER_ARG_FUNCTIONS = {'add_panel_args': ('eval_panel.py', 'add_panel_args')}


def _helper_specs(filename, function_name):
    path = os.path.join(REPO_ROOT, filename)
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            tree = ast.parse(handle.read(), filename=path)
    except (OSError, SyntaxError):
        return []
    specs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for call in ast.walk(node):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and call.func.attr == 'add_argument'):
                    spec = _spec_from_call(call)
                    if spec is not None:
                        specs.append(spec)
    return specs


def _spec_from_call(node):
    ''' ArgSpec for one parser.add_argument(...) call, or None. '''
    # Longest option string wins ('-v', '--verbose' -> '--verbose')
    flags = [v for v in (_safe_eval(a) for a in node.args) if isinstance(v, str)]
    flags = [f for f in flags if f.startswith('-')]
    if not flags:
        return None  # positional argument; none of these scripts use one
    flag = max(flags, key=len)

    spec = ArgSpec(flag=flag)
    for kw in node.keywords:
        if kw.arg == 'type':
            if isinstance(kw.value, ast.Name):
                spec.type_name = kw.value.id
        elif kw.arg == 'default':
            spec.default = _safe_eval(kw.value)
            if spec.default is _UNKNOWN:
                spec.raw_default = _node_source(kw.value)
        elif kw.arg == 'nargs':
            spec.nargs = _safe_eval(kw.value)
        elif kw.arg == 'choices':
            value = _safe_eval(kw.value)
            if isinstance(value, (list, tuple)):
                spec.choices = [str(v) for v in value]
        elif kw.arg == 'help':
            value = _safe_eval(kw.value)
            if isinstance(value, str):
                spec.help_text = value
        elif kw.arg == 'action':
            value = _safe_eval(kw.value)
            if isinstance(value, str):
                spec.action = value
    return spec


# argparse scripts that are not training scripts (tune.py has its own GUI)
NOT_TRAINING_SCRIPTS = {'tune.py', 'tune_gui.py', 'exploitability.py'}


def discover_scripts(root):
    ''' Find the scripts in `root` that expose an argparse command line. '''
    found = []
    for name in sorted(os.listdir(root)):
        if not name.endswith('.py') or name == os.path.basename(__file__):
            continue
        if name in NOT_TRAINING_SCRIPTS or name.startswith('test_'):
            continue
        path = os.path.join(root, name)
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                source = handle.read()
        except OSError:
            continue
        if 'ArgumentParser' in source and 'add_argument' in source:
            found.append(name)
    return found


def value_to_argv(spec, text):
    ''' Convert one form field into argv items.

    An empty field contributes nothing, so the script's own default applies.
    That matters for options like --log_dir and --device, whose None default
    triggers logic inside the script (dynamic log directory, device
    auto-detection) that we must not override with an empty string.

    Returns:
        (list) argv items, and raises ValueError if the text does not parse.
    '''
    text = text.strip()

    if spec.is_flag:
        # store_true / store_false: presence is the value
        return [spec.flag] if text in ('1', 'True', 'true') else []

    if spec.is_bool:
        # argparse's type=bool calls bool() on the *string*, so bool('False')
        # is True. bool('') is False, so an empty string is the only false
        # value that works for both type=bool and type=str2bool.
        return [spec.flag, '1'] if text in ('1', 'True', 'true') else [spec.flag, '']

    if not text:
        return []

    if spec.is_list:
        tokens = text.split()
        for token in tokens:
            _check_scalar(spec, token)
        return [spec.flag] + tokens

    _check_scalar(spec, text)
    return [spec.flag, text]


def _check_scalar(spec, token):
    if spec.type_name == 'int':
        int(token)
    elif spec.type_name == 'float':
        float(token)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

class Tooltip(object):
    ''' Minimal hover tooltip, used to show each option's help= text. '''

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.window = None
        widget.bind('<Enter>', self._show, add='+')
        widget.bind('<Leave>', self._hide, add='+')

    def _show(self, _event=None):
        if self.window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry('+%d+%d' % (x, y))
        label = tk.Label(
            self.window, text=self.text, justify='left', background='#ffffe0',
            relief='solid', borderwidth=1, wraplength=460, font=('Segoe UI', 8))
        label.pack(ipadx=4, ipady=2)

    def _hide(self, _event=None):
        if self.window:
            self.window.destroy()
            self.window = None


class ScrollableFrame(ttk.Frame):
    ''' A frame whose contents scroll vertically. '''

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)

        self.inner = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor='nw')

        self.inner.bind('<Configure>', self._on_inner_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.canvas.bind('<Enter>', lambda _e: self.canvas.bind_all('<MouseWheel>', self._on_wheel))
        self.canvas.bind('<Leave>', lambda _e: self.canvas.unbind_all('<MouseWheel>'))

    def _on_inner_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._window, width=event.width)

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), 'units')


# --------------------------------------------------------------------------- #
# The application
# --------------------------------------------------------------------------- #

class TrainerGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('RLCard Training Launcher')
        self.geometry('1080x820')
        self.minsize(820, 600)

        self.scripts = discover_scripts(REPO_ROOT)
        if not self.scripts:
            messagebox.showerror('No scripts', 'No argparse-based scripts found in\n' + REPO_ROOT)
            self.destroy()
            return

        self.specs = []
        self.fields = {}          # flag -> tk.StringVar
        self.proc = None
        self.queue = queue.Queue()
        self.reader = None
        self.start_time = None
        self._live_len = 0

        self.script_var = tk.StringVar(value=self._default_script())
        self.interp_var = tk.StringVar(value=sys.executable)
        self.autoscroll_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value='Idle')

        self._build_ui()
        self._load_script()
        self.after(80, self._drain_output)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    def _default_script(self):
        for preferred in ('limit_ppo_selfplay.py',):
            if preferred in self.scripts:
                return preferred
        return self.scripts[0]

    # ---------------------------------------------------------------- UI --- #

    def _build_ui(self):
        top = ttk.Frame(self, padding=(8, 8, 8, 4))
        top.pack(side='top', fill='x')

        ttk.Label(top, text='Script').grid(row=0, column=0, sticky='w', padx=(0, 6))
        script_box = ttk.Combobox(top, textvariable=self.script_var,
                                  values=self.scripts, state='readonly', width=44)
        script_box.grid(row=0, column=1, sticky='w')
        script_box.bind('<<ComboboxSelected>>', lambda _e: self._load_script())

        self.desc_label = ttk.Label(top, text='', foreground='#555')
        self.desc_label.grid(row=0, column=2, sticky='w', padx=10)

        ttk.Label(top, text='Interpreter').grid(row=1, column=0, sticky='w', pady=(6, 0), padx=(0, 6))
        interp_entry = ttk.Entry(top, textvariable=self.interp_var, width=70)
        interp_entry.grid(row=1, column=1, columnspan=2, sticky='we', pady=(6, 0))
        Tooltip(interp_entry, 'Python executable used to run the script. It needs rlcard, '
                              'torch and matplotlib installed -- use Check to confirm.')

        interp_buttons = ttk.Frame(top)
        interp_buttons.grid(row=1, column=3, sticky='w', padx=6, pady=(6, 0))
        ttk.Button(interp_buttons, text='Browse...', command=self._browse_interpreter).pack(side='left')
        ttk.Button(interp_buttons, text='Check', command=self._check_interpreter).pack(side='left', padx=(4, 0))

        top.columnconfigure(2, weight=1)

        panes = ttk.PanedWindow(self, orient='vertical')
        panes.pack(side='top', fill='both', expand=True, padx=8, pady=4)

        # --- hyperparameter form ---
        form_outer = ttk.LabelFrame(panes, text='Hyperparameters', padding=6)
        self.form = ScrollableFrame(form_outer)
        self.form.pack(fill='both', expand=True)
        panes.add(form_outer, weight=3)

        # --- command preview + buttons ---
        mid = ttk.Frame(panes)
        cmd_frame = ttk.LabelFrame(mid, text='Command', padding=6)
        cmd_frame.pack(side='top', fill='x')
        self.cmd_text = tk.Text(cmd_frame, height=3, wrap='word', font=('Consolas', 9))
        self.cmd_text.pack(side='left', fill='both', expand=True)
        self.cmd_text.configure(state='disabled')

        buttons = ttk.Frame(mid, padding=(0, 6, 0, 0))
        buttons.pack(side='top', fill='x')
        self.run_button = ttk.Button(buttons, text='Run', command=self._run)
        self.run_button.pack(side='left')
        self.stop_button = ttk.Button(buttons, text='Stop', command=self._stop, state='disabled')
        self.stop_button.pack(side='left', padx=(6, 0))
        ttk.Separator(buttons, orient='vertical').pack(side='left', fill='y', padx=10)
        ttk.Button(buttons, text='Reset defaults', command=self._reset_defaults).pack(side='left')
        ttk.Button(buttons, text='Copy command', command=self._copy_command).pack(side='left', padx=(6, 0))
        ttk.Separator(buttons, orient='vertical').pack(side='left', fill='y', padx=10)
        ttk.Button(buttons, text='Save preset', command=self._save_preset).pack(side='left')
        ttk.Button(buttons, text='Load preset', command=self._load_preset).pack(side='left', padx=(6, 0))
        ttk.Button(buttons, text='Open experiments', command=self._open_experiments).pack(side='right')
        panes.add(mid, weight=0)

        # --- output log ---
        log_frame = ttk.LabelFrame(panes, text='Output', padding=6)
        log_toolbar = ttk.Frame(log_frame)
        log_toolbar.pack(side='top', fill='x', pady=(0, 4))
        ttk.Checkbutton(log_toolbar, text='Auto-scroll',
                        variable=self.autoscroll_var).pack(side='left')
        ttk.Button(log_toolbar, text='Clear', command=self._clear_log).pack(side='left', padx=(10, 0))
        ttk.Button(log_toolbar, text='Save log...', command=self._save_log).pack(side='left', padx=(6, 0))

        log_inner = ttk.Frame(log_frame)
        log_inner.pack(side='top', fill='both', expand=True)
        log_scroll = ttk.Scrollbar(log_inner, orient='vertical')
        self.log = tk.Text(log_inner, wrap='none', font=('Consolas', 9),
                           background='#1e1e1e', foreground='#d4d4d4',
                           insertbackground='#d4d4d4', yscrollcommand=log_scroll.set)
        log_scroll.configure(command=self.log.yview)
        log_scroll.pack(side='right', fill='y')
        self.log.pack(side='left', fill='both', expand=True)
        self.log.tag_configure('meta', foreground='#4ec9b0')
        self.log.tag_configure('error', foreground='#f48771')
        self.log.mark_set('live_start', '1.0')
        self.log.mark_gravity('live_start', 'left')
        self.log.configure(state='disabled')
        panes.add(log_frame, weight=3)

        status = ttk.Frame(self, padding=(8, 2, 8, 6))
        status.pack(side='bottom', fill='x')
        ttk.Label(status, textvariable=self.status_var).pack(side='left')

    # ------------------------------------------------------------ script --- #

    def _load_script(self):
        ''' Re-read the selected script's argparse definition and rebuild the form. '''
        path = os.path.join(REPO_ROOT, self.script_var.get())
        try:
            description, specs = parse_script(path)
        except (OSError, SyntaxError) as exc:
            messagebox.showerror('Cannot read script', '%s\n\n%s' % (path, exc))
            return

        self.specs = specs
        self.desc_label.configure(text=description)

        for child in self.form.inner.winfo_children():
            child.destroy()
        self.fields = {}

        if not specs:
            ttk.Label(self.form.inner,
                      text='This script exposes no command-line options.').grid(
                          row=0, column=0, sticky='w', padx=4, pady=4)
            self._update_command()
            return

        grid = self.form.inner
        for index, spec in enumerate(specs):
            row, col = index // 2, (index % 2) * 2

            label = ttk.Label(grid, text=spec.name)
            label.grid(row=row, column=col, sticky='w', padx=(4, 6), pady=3)

            var = tk.StringVar(value=spec.default_as_text())
            var.trace_add('write', lambda *_a: self._update_command())
            self.fields[spec.flag] = var

            if spec.choices:
                widget = ttk.Combobox(grid, textvariable=var, values=spec.choices,
                                      state='readonly', width=22)
            elif spec.is_bool or spec.is_flag:
                var.set('1' if spec.default in (True, 1) else '')
                widget = ttk.Checkbutton(grid, variable=var, onvalue='1', offvalue='')
            else:
                widget = ttk.Entry(grid, textvariable=var, width=24)

            widget.grid(row=row, column=col + 1, sticky='we', padx=(0, 14), pady=3)

            tip = spec.help_text or '(no help text in the script)'
            details = ['type: %s' % (spec.type_name or 'str')]
            if spec.is_list:
                details.append('several values, space separated')
            if spec.default is _UNKNOWN:
                details.append('default in script: %s (left blank here)' % spec.raw_default)
            elif spec.default is None:
                details.append('default: None -- leave blank to let the script decide')
            else:
                details.append('default: %s' % spec.default_as_text())
            if not spec.is_bool and not spec.is_flag:
                details.append('blank = omit the flag, script default applies')
            Tooltip(label, tip + '\n\n' + '\n'.join(details))
            Tooltip(widget, tip)

        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)
        self._update_command()

    def _reset_defaults(self):
        for spec in self.specs:
            var = self.fields.get(spec.flag)
            if var is None:
                continue
            if spec.is_bool or spec.is_flag:
                var.set('1' if spec.default in (True, 1) else '')
            else:
                var.set(spec.default_as_text())

    # ----------------------------------------------------------- command --- #

    def build_command(self):
        ''' Assemble the argv for the current form state.

        Returns:
            (tuple) (argv (list), errors (list of str))
        '''
        interpreter = self.interp_var.get().strip() or sys.executable
        argv = [interpreter, '-u', os.path.join(REPO_ROOT, self.script_var.get())]
        errors = []

        for spec in self.specs:
            var = self.fields.get(spec.flag)
            if var is None:
                continue
            try:
                argv.extend(value_to_argv(spec, var.get()))
            except ValueError:
                errors.append('%s: %r is not a valid %s'
                              % (spec.name, var.get().strip(), spec.type_name))
        return argv, errors

    def _update_command(self):
        argv, errors = self.build_command()
        self.cmd_text.configure(state='normal')
        self.cmd_text.delete('1.0', 'end')
        self.cmd_text.insert('1.0', ' '.join(_quote(a) for a in argv))
        if errors:
            self.cmd_text.insert('end', '\n! ' + '; '.join(errors))
        self.cmd_text.configure(state='disabled')

    def _copy_command(self):
        argv, _ = self.build_command()
        self.clipboard_clear()
        self.clipboard_append(' '.join(_quote(a) for a in argv))
        self.status_var.set('Command copied to clipboard')

    # --------------------------------------------------------- execution --- #

    def _run(self):
        if self.proc is not None:
            messagebox.showinfo('Already running', 'A training run is already in progress.')
            return

        argv, errors = self.build_command()
        if errors:
            messagebox.showerror('Invalid values', '\n'.join(errors))
            return
        if not os.path.isfile(argv[0]):
            messagebox.showerror('Interpreter not found', argv[0])
            return

        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'

        creationflags = 0
        if os.name == 'nt':
            creationflags = subprocess.CREATE_NO_WINDOW

        try:
            self.proc = subprocess.Popen(
                argv,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                bufsize=0,
                env=env,
                creationflags=creationflags,
            )
        except OSError as exc:
            messagebox.showerror('Could not start process', str(exc))
            self.proc = None
            return

        self._append_meta('$ ' + ' '.join(_quote(a) for a in argv))
        self._append_meta('--- started (pid %d) in %s ---' % (self.proc.pid, REPO_ROOT))

        self.start_time = time.time()
        self.reader = threading.Thread(target=self._read_output,
                                       args=(self.proc, self.queue), daemon=True)
        self.reader.start()

        self.run_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self._tick_status()

    def _read_output(self, proc, out_queue):
        ''' Stream the child's output, reproducing terminal carriage returns.

        The training scripts print progress with print('\\r...', end=''), so a
        plain readline() would stall until the next newline. Reading raw bytes
        and handling '\\r' ourselves keeps those updates live and stops them
        from piling up thousands of near-identical lines in the log.
        '''
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        fd = proc.stdout.fileno()
        current = ''
        # A '\r' only means "overwrite this line" when it is NOT followed by
        # '\n'. On Windows every newline reaches us as '\r\n', and the two
        # bytes can land in different reads, so the pending state has to
        # survive across chunks.
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
                        out_queue.put(('commit', current))
                        current = ''
                        continue
                    # A bare '\r': the line so far is overwritten from here on
                    out_queue.put(('live', current))
                    current = ''

                if char == '\r':
                    pending_cr = True
                elif char == '\n':
                    out_queue.put(('commit', current))
                    current = ''
                elif char != '\x00':
                    current += char

            out_queue.put(('live', current))

        if current:
            out_queue.put(('commit', current))
        out_queue.put(('eof', proc.wait()))

    def _drain_output(self):
        ''' Apply whatever the reader thread has produced. Runs on the UI thread. '''
        commits = []
        live = None
        exit_code = None
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == 'commit':
                    commits.append(payload)
                elif kind == 'live':
                    live = payload
                elif kind == 'eof':
                    exit_code = payload
        except queue.Empty:
            pass

        if commits or live is not None:
            dropped = 0
            if len(commits) > MAX_LINES_PER_TICK:
                dropped = len(commits) - MAX_LINES_PER_TICK
                commits = commits[-MAX_LINES_PER_TICK:]
            self._render(commits, live, dropped)

        if exit_code is not None:
            self._on_finished(exit_code)

        self.after(80, self._drain_output)

    def _render(self, commits, live, dropped=0):
        self.log.configure(state='normal')

        # Drop the previous unterminated tail before committing whole lines
        if self._live_len:
            self.log.delete('live_start', 'end-1c')
            self._live_len = 0

        if dropped:
            self.log.insert('end-1c', '... %d line(s) skipped to keep up ...\n' % dropped, 'meta')
        for line in commits:
            self.log.insert('end-1c', line + '\n')

        total = int(self.log.index('end-1c').split('.')[0])
        if total > MAX_LOG_LINES:
            self.log.delete('1.0', '%d.0' % (total - MAX_LOG_LINES + 1))

        self.log.mark_set('live_start', 'end-1c')
        self.log.mark_gravity('live_start', 'left')
        if live:
            self.log.insert('end-1c', live)
            self._live_len = len(live)

        self.log.configure(state='disabled')
        if self.autoscroll_var.get():
            self.log.see('end')

    def _append_meta(self, text, tag='meta'):
        self.log.configure(state='normal')
        if self._live_len:
            self.log.delete('live_start', 'end-1c')
            self._live_len = 0
        self.log.insert('end-1c', text + '\n', tag)
        self.log.mark_set('live_start', 'end-1c')
        self.log.mark_gravity('live_start', 'left')
        self.log.configure(state='disabled')
        if self.autoscroll_var.get():
            self.log.see('end')

    def _stop(self):
        if self.proc is None:
            return
        self._append_meta('--- stop requested ---')
        try:
            self.proc.terminate()
        except OSError:
            pass

        def force_kill(proc):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass

        threading.Thread(target=force_kill, args=(self.proc,), daemon=True).start()

    def _on_finished(self, exit_code):
        elapsed = time.time() - (self.start_time or time.time())
        if exit_code == 0:
            self._append_meta('--- finished in %s (exit 0) ---' % _fmt_elapsed(elapsed))
            self.status_var.set('Finished in %s' % _fmt_elapsed(elapsed))
        else:
            self._append_meta('--- exited with code %s after %s ---'
                              % (exit_code, _fmt_elapsed(elapsed)), 'error')
            self.status_var.set('Exited with code %s after %s' % (exit_code, _fmt_elapsed(elapsed)))
        self.proc = None
        self.reader = None
        self.start_time = None
        self.run_button.configure(state='normal')
        self.stop_button.configure(state='disabled')

    def _tick_status(self):
        if self.proc is None:
            return
        elapsed = time.time() - (self.start_time or time.time())
        self.status_var.set('Running %s (pid %d) -- %s elapsed'
                            % (self.script_var.get(), self.proc.pid, _fmt_elapsed(elapsed)))
        self.after(1000, self._tick_status)

    # ----------------------------------------------------------- presets --- #

    def _current_values(self):
        return {
            'script': self.script_var.get(),
            'interpreter': self.interp_var.get(),
            'values': {flag: var.get() for flag, var in self.fields.items()},
        }

    def _save_preset(self):
        os.makedirs(PRESET_DIR, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title='Save preset', initialdir=PRESET_DIR, defaultextension='.json',
            filetypes=[('JSON', '*.json')],
            initialfile=os.path.splitext(self.script_var.get())[0] + '.json')
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(self._current_values(), handle, indent=2)
        self.status_var.set('Preset saved to ' + path)

    def _load_preset(self):
        path = filedialog.askopenfilename(
            title='Load preset',
            initialdir=PRESET_DIR if os.path.isdir(PRESET_DIR) else REPO_ROOT,
            filetypes=[('JSON', '*.json')])
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            messagebox.showerror('Cannot load preset', str(exc))
            return

        script = data.get('script')
        if script in self.scripts and script != self.script_var.get():
            self.script_var.set(script)
            self._load_script()
        if data.get('interpreter'):
            self.interp_var.set(data['interpreter'])

        unknown = []
        for flag, value in (data.get('values') or {}).items():
            if flag in self.fields:
                self.fields[flag].set(value)
            else:
                unknown.append(flag)
        message = 'Preset loaded from ' + os.path.basename(path)
        if unknown:
            message += ' (%d option(s) not in this script: %s)' % (
                len(unknown), ', '.join(sorted(unknown)))
        self.status_var.set(message)

    # ------------------------------------------------------------- misc --- #

    def _browse_interpreter(self):
        path = filedialog.askopenfilename(
            title='Select Python interpreter',
            filetypes=[('Python executable', 'python*.exe'), ('All files', '*.*')]
            if os.name == 'nt' else [('All files', '*.*')])
        if path:
            self.interp_var.set(path)

    def _check_interpreter(self):
        interpreter = self.interp_var.get().strip() or sys.executable
        if not os.path.isfile(interpreter):
            messagebox.showerror('Interpreter not found', interpreter)
            return
        probe = (
            'import sys; print("python", sys.version.split()[0])\n'
            'for m in ("rlcard", "torch", "numpy", "matplotlib"):\n'
            '    try:\n'
            '        mod = __import__(m)\n'
            '        print("  OK  ", m, getattr(mod, "__version__", ""))\n'
            '    except Exception as exc:\n'
            '        print("  MISS", m, "->", exc)\n'
        )
        try:
            result = subprocess.run([interpreter, '-c', probe], capture_output=True,
                                    text=True, timeout=120, cwd=REPO_ROOT)
        except (OSError, subprocess.TimeoutExpired) as exc:
            messagebox.showerror('Check failed', str(exc))
            return
        report = (result.stdout or '') + (result.stderr or '')
        ok = 'MISS' not in report
        self._append_meta('--- interpreter check: %s ---' % interpreter)
        for line in report.splitlines():
            self._append_meta(line, 'meta' if 'MISS' not in line else 'error')
        self.status_var.set('Interpreter OK' if ok else 'Interpreter is missing packages -- see Output')

    def _clear_log(self):
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.mark_set('live_start', '1.0')
        self.log.mark_gravity('live_start', 'left')
        self.log.configure(state='disabled')
        self._live_len = 0

    def _save_log(self):
        path = filedialog.asksaveasfilename(
            title='Save log', defaultextension='.txt', filetypes=[('Text', '*.txt')])
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(self.log.get('1.0', 'end-1c'))
        self.status_var.set('Log saved to ' + path)

    def _open_experiments(self):
        target = os.path.join(REPO_ROOT, 'experiments')
        log_dir = self.fields.get('--log_dir')
        if log_dir and log_dir.get().strip():
            candidate = os.path.join(REPO_ROOT, log_dir.get().strip())
            if os.path.isdir(candidate):
                target = candidate
        os.makedirs(target, exist_ok=True)
        try:
            if os.name == 'nt':
                os.startfile(target)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', target])
            else:
                subprocess.Popen(['xdg-open', target])
        except OSError as exc:
            messagebox.showerror('Could not open folder', str(exc))

    def _on_close(self):
        if self.proc is not None:
            if not messagebox.askyesno(
                    'Training in progress',
                    'A run is still going. Stopping it now means the script will not reach '
                    'its final save_checkpoint() or plot_curve() call.\n\nStop it and quit?'):
                return
            try:
                self.proc.kill()
            except OSError:
                pass
        self.destroy()


def _quote(token):
    return '"%s"' % token if (' ' in token or token == '') else token


def _fmt_elapsed(seconds):
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return '%dh %02dm %02ds' % (hours, minutes, secs)
    if minutes:
        return '%dm %02ds' % (minutes, secs)
    return '%ds' % secs


if __name__ == '__main__':
    TrainerGUI().mainloop()
