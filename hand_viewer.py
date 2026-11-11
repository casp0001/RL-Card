''' Visualises analysed example hands (hand_analysis.py output) with the
mistake measure of every decision.

  html     one self-contained page with every *.analysis.json of a folder:
           pick an agent, step through its hands street by street; each
           decision shows the action probabilities, the value of every
           action and the EV loss (Leduc: also the equilibrium's mix)
  figure   one hand as a static figure for the thesis (PDF / PNG)

    python hand_viewer.py --dir experiments/final/hands --out experiments/final/hands/hands.html
    python hand_viewer.py figure --file experiments/final/hands/deepcfr_vs_tag.analysis.json --hand 0 \
        --out experiments/final/hands/deepcfr_hand0.pdf
'''

import os
import sys
import glob
import json
import argparse

ACTION_ORDER = ['fold', 'check', 'call', 'raise']
# action colours (light / dark): validated categorical slots blue, aqua, red, and a neutral grey for fold
ACTION_COLOR = {'fold': ('#898781', '#898781'), 'check': ('#1baf7a', '#199e70'),
                'call': ('#2a78d6', '#3987e5'), 'raise': ('#e34948', '#e66767')}
VERB = {'fold': 'folds', 'check': 'checks', 'call': 'calls', 'raise': 'raises'}


def load(folder):
    data = []
    for path in sorted(glob.glob(os.path.join(folder, '*.analysis.json'))):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        r['file'] = os.path.basename(path)
        r['label'] = r['hands'][0]['names']['agent'] if r['hands'] else r['file']
        data.append(r)
    return data


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Example Hands</title>
<style>
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--axis:#c3c2b7;
--fold:#898781;--check:#1baf7a;--call:#2a78d6;--raise:#e34948;--loss:#1c5cab;--chip:#f0efec;--card:#ffffff;--red:#c0392b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#898781;--line:#2c2c2a;--axis:#383835;--check:#199e70;--call:#3987e5;--raise:#e66767;--loss:#86b6ef;--chip:#2c2c2a;--card:#f0efec;--red:#c0392b}}
:root[data-theme="dark"]{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--axis:#383835;
--check:#199e70;--call:#3987e5;--raise:#e66767;--loss:#86b6ef;--chip:#2c2c2a;--card:#f0efec;--red:#c0392b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
header{padding:20px 16px 8px;max-width:1100px;margin:0 auto}h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--ink2);max-width:820px}.controls{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 4px}
select,button{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--axis);border-radius:6px;padding:6px 10px}
button{cursor:pointer}button[aria-pressed="true"]{border-color:var(--ink);font-weight:600}
main{max-width:1100px;margin:0 auto;padding:0 16px 40px}
.summary{color:var(--ink2);margin:6px 0 14px}
.hand{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}
.hhead{display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px;margin-bottom:8px}
.cat{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:12px;color:var(--ink2)}
.who{display:flex;align-items:center;gap:6px}.res{margin-left:auto;font-weight:600}
.cards{display:inline-flex;gap:3px}.c{display:inline-block;min-width:26px;padding:2px 4px;border-radius:4px;background:var(--card);
color:#111;border:1px solid var(--axis);text-align:center;font-weight:700;font-size:13px}.c.r{color:var(--red)}
.street{margin-top:10px;padding-top:8px;border-top:1px solid var(--line);display:flex;gap:10px;align-items:center;color:var(--ink2);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.dec{display:grid;grid-template-columns:150px 1fr 300px;gap:4px 14px;align-items:center;padding:6px 0}
.dec.opp{opacity:.8}.name{font-weight:600}.act{color:var(--ink2)}
.bar{display:flex;height:14px;border-radius:4px;overflow:hidden;background:var(--line);gap:2px}
.bar span{display:block;height:100%}.bar.eq{height:7px;margin-top:3px;opacity:.75}
.legend{font-size:11px;color:var(--ink2);margin-top:2px}
.vals{font-size:12px;color:var(--ink2);font-variant-numeric:tabular-nums}.vals b{color:var(--ink)}
.loss{display:flex;align-items:center;gap:8px;font-variant-numeric:tabular-nums}
.lbar{height:8px;background:var(--loss);border-radius:2px;min-width:1px}
.note{font-size:11px;color:var(--muted)}
.keys{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:6px 0}
.keys i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
@media (max-width:720px){.dec{grid-template-columns:1fr}.res{margin-left:0}}
</style></head><body>
<header><h1>Example hands with the mistake measure</h1>
<p class="sub">Every decision of the agent shows its action probabilities, the expected value Q of each legal action
(big blinds, for the whole hand, so a fold is worth minus the chips already invested) and the <b>EV loss</b>:
value of the best action minus the value of the agent's own mix. <b>Limit hold'em</b>: against the actual opponent
(TAG), estimated by playing every action out against the opponent's likely holdings (Bayes' rule on its actions);
a deliberately mixing agent has some EV loss against a fixed opponent by design, so this measures how well it
exploits that opponent. <b>Leduc</b>: exact, against the CFR+ equilibrium; the thin bar is the equilibrium's own mix.
Each Q assumes both players continue as they actually play, so when the agent's later play is poor even a normally
dominated action (folding when a check is free) can come out best. Limit values are estimates (&plusmn; one standard
error; the EV loss is cross-fitted, so it can be slightly negative when the actions are about equally good).</p>
<div class="controls"><select id="agent" aria-label="Agent"></select>
<button id="theme" type="button" aria-label="Toggle dark mode">Theme</button></div>
<div class="keys" id="keys"></div></header>
<main><div class="summary" id="summary"></div><div id="hands"></div></main>
<script>
const DATA = __DATA__;
const ORDER = ['fold','check','call','raise'], VERB = {fold:'folds',check:'checks',call:'calls',raise:'raises'};
const sel = document.getElementById('agent');
DATA.forEach((r, i) => { const o = document.createElement('option'); o.value = i;
  o.textContent = (r.game === 'leduc' ? 'Leduc: ' : "Limit: ") + r.label + ' vs ' + (r.game === 'leduc' ? 'CFR+' : r.opponent.toUpperCase());
  sel.appendChild(o); });
try { const s = localStorage.getItem('hv-agent'); if (s !== null && DATA[s]) sel.value = s; } catch (e) {}
document.getElementById('keys').innerHTML = ORDER.map(a => `<span><i style="background:var(--${a})"></i>${a}</span>`).join('');
document.getElementById('theme').onclick = () => { const r = document.documentElement;
  const dark = r.dataset.theme ? r.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  r.dataset.theme = dark ? 'light' : 'dark'; };
function card(c){ if (c.length === 1) return `<span class="c">${c}</span>`;
  const suit = {S:'&#9824;',H:'&#9829;',D:'&#9830;',C:'&#9827;'}[c[0]], red = 'HD'.includes(c[0]);
  return `<span class="c${red ? ' r' : ''}">${c[1] === 'T' ? '10' : c[1]}${suit}</span>`; }
const cards = cs => `<span class="cards">${cs.map(card).join('')}</span>`;
const f = (v, d=2) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
function bar(probs, legal, cls=''){ return `<div class="bar ${cls}">` + ORDER.filter(a => legal.includes(a)).map(a => {
  const p = probs[a] || 0; return p > 0.0005 ? `<span style="width:${100*p}%;background:var(--${a})" title="${a} ${(100*p).toFixed(1)}%"></span>` : ''; }).join('') + '</div>'; }
function render(){
  const r = DATA[sel.value]; try { localStorage.setItem('hv-agent', sel.value); } catch (e) {}
  const losses = r.hands.flatMap(h => h.decisions.filter(d => d.analysis && d.analysis.ev_loss !== undefined).map(d => d.analysis.ev_loss));
  const maxLoss = Math.max(0.5, ...losses);
  let s = `${r.hands.length} hands, ${losses.length} analysed decisions; mean EV loss ${(losses.reduce((a,b)=>a+b,0)/Math.max(1,losses.length)).toFixed(3)} BB per decision in these hands.`;
  if (r.summary) s += ` Over all ${r.summary.decisions} decisions of ${r.summary.deals} hands against CFR+: mean EV loss ${r.summary.mean_ev_loss.toFixed(3)} BB, ${(100*r.summary['share_with_loss_over_0.05bb']).toFixed(0)}% of decisions lose more than 0.05 BB.`;
  document.getElementById('summary').textContent = s;
  document.getElementById('hands').innerHTML = r.hands.map(h => {
    const N = h.names; let street = null, html = '';
    html += `<div class="hhead"><span class="cat">${h.category || ''}</span>
      <span class="who"><span class="name">${N.agent}</span> ${h.positions.agent} ${cards(h.holes.agent)}</span>
      <span class="who"><span class="name">${N.opponent}</span> ${h.positions.opponent} ${cards(h.holes.opponent)}</span>
      <span class="res">${N.agent} ${f(h.payoff_bb.agent, 1)} BB</span></div>`;
    for (const d of h.decisions){
      if (d.street !== street){ street = d.street;
        html += `<div class="street"><span>${street}</span>${d.board && d.board.length ? cards(d.board) : ''}${d.pot_bb !== undefined ? `<span>pot ${d.pot_bb.toFixed(1)} BB</span>` : ''}</div>`; }
      const me = d.role === 'agent', an = d.analysis;
      let mid = d.probs ? bar(d.probs, d.legal) : '';
      if (an && an.equilibrium_probs) mid += bar(an.equilibrium_probs, d.legal, 'eq');
      if (d.probs) mid += `<div class="legend">${d.legal.filter(a => (d.probs[a]||0) > 0.005).map(a => `${a} ${(100*d.probs[a]).toFixed(0)}%`).join(' · ')}${an && an.equilibrium_probs ? ' &nbsp;|&nbsp; CFR+: ' + d.legal.filter(a => an.equilibrium_probs[a] > 0.005).map(a => `${a} ${(100*an.equilibrium_probs[a]).toFixed(0)}%`).join(' · ') : ''}</div>`;
      let right = '';
      if (me && an){
        const vals = d.legal.map(a => (a === an.best ? '<b>' : '') + `${a} ${f(an.values[a])}` + (an.values_se[a] > 0 ? `±${an.values_se[a].toFixed(2)}` : '') + (a === an.best ? '</b>' : '')).join(' · ');
        const w = Math.max(0, Math.min(100, 100 * an.ev_loss / maxLoss));
        right = `<div class="loss"><span>EV loss <b>${an.ev_loss.toFixed(3)}</b>${an.ev_loss_se > 0 ? ' ± ' + an.ev_loss_se.toFixed(3) : ''} BB</span><span class="lbar" style="width:${w}px"></span></div><div class="vals">Q: ${vals}</div>`
          + (an.opponent_reach !== undefined && an.opponent_reach < 1e-3 ? `<div class="note">CFR+ almost never plays into this state (reach ${an.opponent_reach.toExponential(1)})</div>` : '')
          + (an.ess !== undefined ? `<div class="note">${an.samples} rollouts per action; opponent range: ${an.candidates} holdings, effective ${an.ess.toFixed(0)}</div>` : '');
      } else if (me){ right = '<div class="note">single legal action</div>'; }
      html += `<div class="dec ${me ? '' : 'opp'}"><div><span class="name">${N[d.role]}</span> <span class="act">${VERB[d.action]}</span></div><div>${mid}</div><div>${right}</div></div>`;
    }
    html += `<div class="street"><span>${h.showdown ? 'showdown' : N[h.folded] + ' folded'}</span>${h.board && h.board.length ? cards(h.board) : ''}</div>`;
    return `<section class="hand" aria-label="hand">${html}</section>`;
  }).join('');
}
sel.onchange = render; render();
</script></body></html>'''


def write_html(folder, out):
    data = load(folder)
    if not data:
        raise SystemExit(f'no *.analysis.json in {folder}')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(PAGE.replace('__DATA__', json.dumps(data).replace('</', '<\\/')))
    print(f'{out}: {len(data)} agents, {sum(len(r["hands"]) for r in data)} hands')


# --------------------------------------------------------------------------- #
# Static figure
# --------------------------------------------------------------------------- #

def _card_text(c):
    if len(c) == 1:
        return c
    return ('10' if c[1] == 'T' else c[1]) + {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}[c[0]]


def write_figure(path, index, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with open(path, encoding='utf-8') as f:
        r = json.load(f)
    h = r['hands'][index]
    rows = h['decisions']
    eq = any(d.get('analysis') and d['analysis'].get('equilibrium_probs') for d in rows)
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(len(rows) + 0.2, -1.4)
    ax.axis('off')
    n = h['names']
    head = (f"{n['agent']} {h['positions']['agent']} [{' '.join(map(_card_text, h['holes']['agent']))}]  vs  "
            f"{n['opponent']} {h['positions']['opponent']} [{' '.join(map(_card_text, h['holes']['opponent']))}]"
            f"   board [{' '.join(map(_card_text, h['board']))}]   result {h['payoff_bb']['agent']:+.1f} BB")
    ax.text(0, -1.0, head, fontsize=9, color='#0b0b0b', va='center')
    ax.text(3.1, -0.4, 'action probabilities' + (' (thin: CFR+ equilibrium)' if eq else ''), fontsize=7,
            color='#52514e', va='center')
    ax.text(7.3, -0.4, 'EV loss (BB); below: value of each action', fontsize=7, color='#52514e', va='center')
    street = None
    for i, d in enumerate(rows):
        y = i + 0.2
        label = (f"{d['street'] if d['street'] != street else '':8s}")
        street = d['street']
        ax.text(0, y, label, fontsize=7, color='#898781', va='center')
        ax.text(1.0, y, f"{n[d['role']]} {VERB[d['action']]}", fontsize=8, va='center',
                color='#0b0b0b' if d['role'] == 'agent' else '#52514e')
        x = 3.1
        for a in ACTION_ORDER:
            if a in d['legal'] and d.get('probs') and d['probs'].get(a, 0) > 0:
                w = 3.8 * d['probs'][a]
                ax.add_patch(plt.Rectangle((x, y - 0.18), w, 0.3, color=ACTION_COLOR[a][0], linewidth=0))
                x += w
        an = d.get('analysis')
        if an and an.get('equilibrium_probs'):
            x = 3.1
            for a in ACTION_ORDER:
                p = an['equilibrium_probs'].get(a, 0)
                if p > 0:
                    ax.add_patch(plt.Rectangle((x, y + 0.16), 3.8 * p, 0.1, color=ACTION_COLOR[a][0], alpha=0.7,
                                               linewidth=0))
                    x += 3.8 * p
        if d['role'] == 'agent' and an:
            se = f" ±{an['ev_loss_se']:.2f}" if an.get('ev_loss_se') else ''
            vals = '  '.join(f"{a} {an['values'][a]:+.2f}" for a in d['legal'])
            ax.text(7.3, y - 0.1, f"{an['ev_loss']:.3f}{se}", fontsize=8, va='center', color='#0b0b0b')
            ax.text(7.3, y + 0.22, vals, fontsize=6, va='center', color='#52514e')
    for k, a in enumerate(ACTION_ORDER):
        ax.add_patch(plt.Rectangle((3.1 + k * 0.95, len(rows) - 0.05), 0.2, 0.2, color=ACTION_COLOR[a][0]))
        ax.text(3.35 + k * 0.95, len(rows) + 0.05, a, fontsize=7, color='#52514e', va='center')
    fig.savefig(out, bbox_inches='tight', dpi=200, facecolor='#fcfcfb')
    print(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', nargs='?', default='html', choices=['html', 'figure'])
    p.add_argument('--dir', default=os.path.join('experiments', 'final', 'hands'))
    p.add_argument('--out', default=None)
    p.add_argument('--file', help='figure: the .analysis.json file')
    p.add_argument('--hand', type=int, default=0, help='figure: index of the hand in that file')
    a = p.parse_args()
    if a.command == 'html':
        write_html(a.dir, a.out or os.path.join(a.dir, 'hands.html'))
    else:
        if not a.file:
            p.error('figure needs --file')
        write_figure(a.file, a.hand, a.out or a.file.replace('.analysis.json', f'_hand{a.hand}.pdf'))


if __name__ == '__main__':
    main()
