''' A Tkinter frame for browsing recorded hands (behaviour_tracker hand logs):
filter + role selection, a hand list, the rendered replay and HTML export.
Used by play_gui.py (review / replay window) and tracker_gui.py. '''

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import behaviour_tracker as bt


class HandBrowser(ttk.Frame):

    def __init__(self, parent, default_role='agent'):
        super().__init__(parent)
        self.hands = []
        self.shown = []
        self.default_role = default_role
        self.role_var = tk.StringVar(value=default_role)
        self.filter_var = tk.StringVar(value='all')
        self.probs_var = tk.BooleanVar(value=True)

        bar = ttk.Frame(self, padding=(0, 0, 0, 6))
        bar.pack(side='top', fill='x')
        ttk.Label(bar, text='Filter').pack(side='left')
        box = ttk.Combobox(bar, textvariable=self.filter_var, values=sorted(bt.FILTERS), state='readonly', width=12)
        box.pack(side='left', padx=(4, 10))
        box.bind('<<ComboboxSelected>>', lambda _e: self.refresh())
        ttk.Label(bar, text='for').pack(side='left')
        self.role_box = ttk.Combobox(bar, textvariable=self.role_var, state='readonly', width=10)
        self.role_box.pack(side='left', padx=(4, 10))
        self.role_box.bind('<<ComboboxSelected>>', lambda _e: self.refresh())
        ttk.Checkbutton(bar, text='show probabilities', variable=self.probs_var,
                        command=self._show).pack(side='left')
        ttk.Button(bar, text='Export shown to HTML...', command=self._export_html).pack(side='right')
        self.info = ttk.Label(bar, text='', foreground='#555')
        self.info.pack(side='left', padx=12)

        panes = ttk.PanedWindow(self, orient='horizontal')
        panes.pack(fill='both', expand=True)
        left = ttk.Frame(panes)
        scroll = ttk.Scrollbar(left, orient='vertical')
        self.listbox = tk.Listbox(left, width=38, font=('Consolas', 9), yscrollcommand=scroll.set,
                                  exportselection=False)
        scroll.configure(command=self.listbox.yview)
        scroll.pack(side='right', fill='y')
        self.listbox.pack(side='left', fill='both', expand=True)
        self.listbox.bind('<<ListboxSelect>>', lambda _e: self._show())
        panes.add(left, weight=1)
        right = ttk.Frame(panes)
        xs = ttk.Scrollbar(right, orient='horizontal')
        ys = ttk.Scrollbar(right, orient='vertical')
        self.text = tk.Text(right, wrap='none', font=('Consolas', 10), xscrollcommand=xs.set,
                            yscrollcommand=ys.set)
        xs.configure(command=self.text.xview)
        ys.configure(command=self.text.yview)
        xs.pack(side='bottom', fill='x')
        ys.pack(side='right', fill='y')
        self.text.pack(side='left', fill='both', expand=True)
        for tag, colour in (('BLUFF', '#b45309'), ('SEMI-BLUFF', '#7c3aed'), ('SLOW-PLAY', '#047857'),
                            ('behind', '#b91c1c')):
            self.text.tag_configure(tag, foreground=colour, font=('Consolas', 10, 'bold'))
        panes.add(right, weight=4)

    def set_hands(self, hands):
        self.hands = list(hands or [])
        roles = sorted(self.hands[0]['holes']) if self.hands else []
        self.role_box.configure(values=roles)
        if roles and self.role_var.get() not in roles:
            self.role_var.set(self.default_role if self.default_role in roles else roles[0])
        self.refresh()

    def refresh(self):
        role = self.role_var.get()
        self.shown = (bt.select_hands(self.hands, self.filter_var.get(), role, len(self.hands))
                      if self.hands else [])
        self.listbox.delete(0, 'end')
        for h in self.shown:
            names = h['names']
            self.listbox.insert('end', f"deal {h['deal']}{'b' if h.get('swap') else ' '} "
                                       f"{h['payoff_bb'][role]:+6.1f} BB  pot {h['pot_bb']:5.1f}"
                                       f"{'  SD' if h['showdown'] else ''}")
        self.info.configure(text=f'{len(self.shown)} of {len(self.hands)} hands' if self.hands else 'no hands')
        self.text.delete('1.0', 'end')
        if self.shown:
            self.listbox.selection_set(0)
            self._show()

    def _show(self):
        sel = self.listbox.curselection()
        self.text.delete('1.0', 'end')
        if not sel or not self.shown:
            return
        text = bt.render_hand(self.shown[sel[0]], show_probs=self.probs_var.get())
        self.text.insert('1.0', text)
        for tag in ('SEMI-BLUFF', 'BLUFF', 'SLOW-PLAY', 'behind'):
            start = '1.0'
            while True:
                pos = self.text.search(tag, start, 'end')
                if not pos:
                    break
                end = f'{pos}+{len(tag)}c'
                # 'BLUFF' inside 'SEMI-BLUFF' keeps the semi-bluff colour
                if not (tag == 'BLUFF' and self.text.get(f'{pos}-5c', pos) == 'SEMI-'):
                    self.text.tag_add(tag, pos, end)
                start = end

    def _export_html(self):
        if not self.shown:
            messagebox.showinfo('Nothing to export', 'No hands match the filter.', parent=self)
            return
        path = filedialog.asksaveasfilename(title='Export hands to HTML', defaultextension='.html',
                                            filetypes=[('HTML', '*.html')], parent=self)
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(bt.render_html(self.shown, f'{self.filter_var.get()} ({self.role_var.get()})'))
            try:
                os.startfile(path)
            except (AttributeError, OSError):
                pass
