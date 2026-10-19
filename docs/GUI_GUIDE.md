# GUI guide

This repository has four graphical front-ends. Each one wraps scripts that also
work from the command line, so anything you do in a GUI you can also script.

| GUI | Start with | Use it to |
|---|---|---|
| [Training launcher](#1-training-launcher-train_guipy) | `python train_gui.py` | run any training script with a form instead of a long command line |
| [Hyper-parameter tuning](#2-hyper-parameter-tuning-tune_guipy) | `python tune_gui.py` | run Optuna studies for DQN, PPO, NFSP and Deep CFR, then pick the best configuration |
| [Behaviour tracker](#3-behaviour-tracker-tracker_guipy) | `python tracker_gui.py` | measure an agent's playing style (bluffing, aggression, ...) and browse its hands |
| [Play against an agent](#4-play-against-an-agent-play_guipy) | `python play_gui.py` | play limit hold'em yourself against any trained agent or rule-based bot |

Run every command from the repository root.

---

## Before you start

### Python interpreter

The training, tuning and tracking GUIs start their work in a **separate Python
process**. That process needs `rlcard`, `torch`, `numpy` and `matplotlib`; the
tuner also needs `optuna` (`scikit-learn` is optional, for parameter
importances). Each of these GUIs has an **Interpreter** field: put the path of
the Python that has these packages there. In the training and tuning GUIs,
**Check** lists which packages that interpreter has.

The GUIs themselves are lighter:

* `train_gui.py` and `tune_gui.py` use only the standard library.
* `tracker_gui.py` needs `rlcard` and `matplotlib`.
* `play_gui.py` loads the agents itself, so it needs everything (`rlcard`, `torch`, ...).

### Agent kinds and checkpoints

Wherever a GUI asks for an agent or opponent, these kinds are available:

| Kind | What it is | Needs a checkpoint? |
|---|---|---|
| `dqn`, `nfsp`, `deepcfr`, `ppo` | a trained agent | yes: the checkpoint file, or the run folder that contains it (`checkpoint_dqn.pt`, `checkpoint_nfsp.pt`, `checkpoint_ppo.pt`, or Deep CFR's `model.ckpt`) |
| `tag` | the "reasonable" rule-based bot: tight-aggressive, driven by hand equity and pot odds | no |
| `nit` | tight-passive variant (folds a lot, never bluffs) | no |
| `lag` | loose-aggressive variant (raises and bluffs a lot) | no |
| `honest` | the old rule bot (weak; kept for comparison) | no |
| `station` | always checks or calls | no |
| `maniac` | always raises | no |
| `random` | uniformly random legal actions | no |

How each trained agent plays when it is evaluated: NFSP uses its average
policy, DQN plays greedily, Deep CFR uses its average-strategy network, and PPO
uses whatever its checkpoint is set to (stochastic by default).

### Observation encodings

Trained agents see the game through one of two encodings (`--obs` in every
training script; the tuner always uses `rich`):

* `rich` (default, 153 values, see `obs_encoding.py`): own hole cards and the
  board separately, street, position (small or big blind), whether you face a
  bet, the raises of every round and who made them, who led the betting on
  earlier streets, the pot and the amount to call.
* `rlcard` (72 values): RLCard's default. Hole and board cards are mixed, and
  it shows neither position nor who raised; kept for comparison and for old checkpoints.

You never choose the encoding when loading an agent: it is read from the
checkpoint's network, and the agent plays correctly in every GUI and tool.

### Units and conventions

* Results are in **mbb/h**: thousandths of a big blind per hand. +1000 mbb/h
  means winning one big blind per hand on average.
* Matches use **duplicate deals**: every deal is played twice with the seats
  swapped, which cancels most of the card luck. Evaluation deals use fixed
  seeds, so every agent faces exactly the same hands.
* RLCard assigns the blinds at random each hand, so "seat 0" and "seat 1"
  carry no meaning; SB and BB do.
* A ± value is one standard error. Two results whose difference is under about
  two standard errors are not clearly different.

---

## 1. Training launcher (`train_gui.py`)

A form-based launcher for the training scripts (`limit_dqn_selfplay_custom.py`,
`limit_ppo_selfplay.py`, the NFSP scripts, `limit_deepcfr_train_custom.py`, ...).
It reads each script's command-line options **from the source code**, so a new
option you add to a script shows up automatically.

### Layout

* **Script**: the training script to run. The description next to it comes
  from the script.
* **Interpreter**: the Python used to run the script (**Browse...**, **Check**).
* **Hyperparameters**: one field per option, with the script's default
  pre-filled. Hover over a name to see its help text, type and default.
  * Leaving a field **blank** omits the option, so the script's own default applies.
  * Several values (layer sizes, for example) are typed separated by spaces: `256 256`.
  * Checkboxes are on/off options.
* **Command**: a live preview of the exact command that will run. Invalid
  values are listed under it.
* **Output**: the script's output as it runs. Progress lines that overwrite
  themselves are shown in place, so they don't flood the log.

### Buttons

| Button | Does |
|---|---|
| Run / Stop | start the script; stop it (asks it to exit, then kills it after 5 s) |
| Reset defaults | refill every field with the script's defaults |
| Copy command | copy the command, e.g. to run it in a terminal |
| Save preset / Load preset | save or restore all field values as JSON in `gui_presets/` |
| Open experiments | open the run's `--log_dir`, or `experiments/` |
| Clear / Save log... | clear the output pane, or save it to a text file |

### Typical use

1. Pick the script, check the interpreter, and adjust the options you care about.
2. Optionally set `--log_dir` so the run doesn't overwrite an older one.
   Several scripts build their folder name from the hyperparameters, so
   re-running the same settings writes to the same folder.
3. **Run**. Results go into the log folder:
   * `panel.csv` and `panel.png`: the evaluation panel over training
     (win rates against random, station, maniac, nit, tag and lag);
   * `performance.csv`: the panel's "min" score, the robustness number used for tuning;
   * the checkpoint(s).

Useful evaluation options in every limit hold'em script:

* `--eval_deals`: duplicate deals per panel member per evaluation (500 gives
  about ±65 mbb/h against the TAG; use more for smoother curves).
* `--eval_panel`: which bots to evaluate against.
* `--evaluate_every`: how often to evaluate.
* `--obs`: the observation encoding (`rich` or `rlcard`; see above).
* `--checkpoint_every`: periodic checkpoints. Use it if you want to measure
  exploitability over training later with `sweep_exploitability.py`.

Closing the window while a run is going asks for confirmation, because the
script won't reach its final save.

---

## 2. Hyper-parameter tuning (`tune_gui.py`)

A front-end for `tune.py`. It uses Optuna's Bayesian search (TPE) to propose
configurations and stops weak trials early (successive halving). Each trial
trains one configuration and reports the evaluation panel's **min** score after
every evaluation. A trial's final score is the mean of its last 3 reports.

### Workflow

1. **Choose the algorithm and a study name.** A new name starts a new study,
   whose files go in `experiments/tune/<study>/`. Typing the name of an
   existing study (or **Browse studies...**) loads it with its settings
   **locked**; *Start* then adds trials to it.
2. **Study settings tab:**

   | Setting | Meaning (default) |
   |---|---|
   | Total trials | trials for the whole study, including pruned and failed ones (40) |
   | Parallel workers | `tune.py` processes working on the study at the same time (2) |
   | Threads / worker | CPU threads per worker; 0 = number of cores / workers |
   | Budget / trial | episodes per trial (Deep CFR: iterations) |
   | Evaluations / trial | panel evaluations per trial; each one is a chance to prune (20) |
   | Deals / evaluation | duplicate deals per panel member (500) |
   | Panel members | which bots make up the score |
   | Training seed | the same for every trial (42) |
   | Pruning | stop trials that fall behind (on) |
   | Min. fraction before pruning | no trial is stopped before this share of its evaluations (0.25), because RL learning curves often cross early |
   | Reduction factor | keep about 1 in this many trials at each pruning step (3) |
   | Random startup trials | random trials before the Bayesian model is used (12) |

   The status bar estimates the buffer memory per worker and compares it with
   the RAM that is free right now. *Start* warns you if it won't fit.
3. **Search space tab:** every tunable parameter with its range (Low / High,
   Log scale) or its choices.
   * Untick **Tune** to fix a parameter at its *Fixed value*.
   * **Targets** shows which script options a parameter sets; for example, one
     "layers" choice can set both of NFSP's networks.
   * **Save space... / Load space...** store a space as JSON;
     **Reset to defaults** goes back to the defaults in `tune_spaces.py`.
4. **Start study.** The workers start a few seconds apart and their output
   streams into the Output pane, prefixed `[w1]`, `[w2]`, ...
5. **Trials tab** (refreshes every 30 s while running, or press **Refresh**):
   number, state (complete / pruned / failed / running), score, how many
   evaluations the trial reached, its last panel min, run time and parameters.
   Click a column header to sort; **double-click** a trial to open its folder
   (its `panel.csv` and `panel.png` show the learning curve).
6. **Run finalists** (top *N*, *seeds*, *deals*): retrains the best completed
   trials with new seeds at the full budget and evaluates each final agent
   with many deals (5000 by default). The summary, including the winner,
   appears under the trial table. This is how to pick each algorithm's final
   configuration; the single best trial is partly luck.
7. **Report + plots** writes figures to `experiments/tune/<study>/plots/`:
   optimisation history, per-trial learning curves, parallel-coordinate and
   slice plots, and parameter importances. It offers to open the folder.

### Things to know

* **Stopping workers** loses the trials they were running. The next *Start*
  marks those as failed; finished trials are kept.
* All workers of a study must use the same settings. That's why an existing
  study is locked. To change settings, use a new study name.
* Several workers can share a study **on one machine** (they share the study's
  journal file). With two PCs, give each PC its own studies, for example by
  splitting the algorithms between them.
* Tuning trials don't save checkpoints (they can be gigabytes for NFSP);
  finalists do.
* The settings fixed for every trial are in `FIXED` in `tune_spaces.py`.
  For example, PPO and NFSP use a discount factor of 1.0 during tuning.

The same operations from a terminal:
`python tune.py run | status | finalists | report --study <name> ...`.

---

## 3. Behaviour tracker (`tracker_gui.py`)

A front-end for `behaviour_tracker.py`. It plays an agent against an opponent
on duplicate deals, records every decision, and computes playing-style
statistics, including how often an agent bluffs.

### Track tab

1. Choose the **Agent** and the **Opponent** (any kind; trained agents need a
   checkpoint via **File...** or **Folder...**), the number of **duplicate
   deals**, and the **output folder** (a dated folder under
   `experiments/behaviour/` is suggested).
   * About 500 deals gives a quick look; use 2000+ for statistics you want to report.
2. **Run match.** The match runs in a separate process, with a progress bar
   and live output; **Stop** cancels it. The Command box shows the exact
   command. Invalid choices, such as a trained agent without a checkpoint,
   are reported there before anything runs.
3. When the match finishes, its results load automatically. **Open hand log...**
   loads any earlier run, via its `hands.jsonl`.

Each run writes `hands.jsonl` (every hand and decision) and
`stats_agent.json` / `stats_opponent.json` to the output folder.

### Stats tab

Both players' statistics side by side, grouped into pre-flop, post-flop,
showdown and bluffing. The **n** column is how many hands or decisions each
number is based on: be careful with small n.

| Statistic | Meaning |
|---|---|
| VPIP / PFR | share of hands where the player voluntarily called / raised pre-flop |
| limp, SB open raise | the small blind's first action: just call vs raise |
| 3-bet | raising when facing one pre-flop raise |
| bet when checked to / fold to bet / raise when bet into | per street (flop, turn, river) |
| aggression factor | post-flop bets and raises divided by calls |
| c-bet | the pre-flop raiser betting the flop when checked to |
| WTSD / W$SD | went to showdown (of hands that saw the flop) / won at showdown |
| bluff rate | how often it bets or raises a **weak** hand |
| bluff / semi-bluff / value share | what its bets and raises are made of |
| slow-play rate | not raising a very strong hand when it could |
| river aggression while behind | river bets made while behind the opponent's actual hand |

Most decision-level rates are **probability-weighted**: they use the agent's
action probabilities rather than the single action it happened to pick, which
makes them much less noisy.

**Thresholds.** A decision's hand class comes from its equity (its chance
of winning against a random hand, counting the cards still to come) and its
*current* hand strength (no more cards):

* `bluff_equity` (0.35): a bet or raise with equity below this, that isn't a draw, is a **bluff**.
* `value_equity` (0.6): a bet or raise with at least this equity is a **value bet**.
* `slowplay_equity` (0.8): checking or calling with at least this equity is a **slow-play**.
* `semi_hs` (0.5) and `semi_potential` (0.08): a flop/turn bet with current
  strength below `semi_hs` but equity at least `semi_potential` higher is a **semi-bluff**.

Edit them and press **Apply**: the statistics, hand tags and filters all
update. **Defaults** restores the values above. State the thresholds you use
in the thesis, since "bluff" has no single standard definition.

### Hands tab

Browse the recorded hands. Pick a **filter** (`bluff`, `semi_bluff`,
`river_bluff`, `river_behind`, `slowplay`, `big_pot`, `big_win`, `big_loss`,
`showdown`, `all`) and the player it applies to (**for**).

* Each hand is shown street by street. Every decision shows its equity, its
  current strength (`hs`), its equity against the opponent's actual hand
  (`vs-hand`) and, optionally, the action probabilities.
* Tags mark notable decisions: `<BLUFF>`, `<SEMI-BLUFF>`, `<SLOW-PLAY>`, and
  `behind` (betting while behind the opponent's actual hand).
* **Export shown to HTML...** writes the selected hands as a readable page
  with coloured suits, handy for thesis figures.

### Compare tab

Build a side-by-side table of several agents:

* **Add current (agent / opponent)**: a player from the loaded match.
* **Add hand log / stats file...**: any `hands.jsonl` or `stats_*.json` file.
* **Remove column / Clear**: edit the table.
* **Export CSV...**: the table, with sample sizes.
* **Chart selected rows**: select statistics (Ctrl/Shift-click) to draw a
  grouped bar chart, which **Save figure...** writes as PNG or PDF.

Columns built from hand logs follow the current thresholds; `stats.json`
columns keep the thresholds they were computed with.

For fair comparisons, track every agent against the **same opponent** with
the **same number of deals**. The deals use fixed seeds, so those matches see
identical cards.

The same from a terminal:
`python behaviour_tracker.py track | replay | stats | compare ...`
(see the top of `behaviour_tracker.py`).

---

## 4. Play against an agent (`play_gui.py`)

Play heads-up limit hold'em against any agent kind.

### Starting

1. Choose the **Opponent**. Trained agents need a **Checkpoint**:
   **Browse...** opens a file picker; cancel it to choose a run folder instead.
2. Press **New session**. A session collects all hands against this opponent
   until you start a new one.

### Playing

* The table shows the opponent at the top, the board and pot in the middle,
  and your cards at the bottom. **SB / BB** marks who posted which blind;
  RLCard assigns them at random each hand. Each player's chips in the pot and
  last action are shown next to their cards.
* Buttons (keys in brackets):

  | Button | When |
  |---|---|
  | Fold (F) | only when facing a bet (it never offers folding when you could check) |
  | Check / Call *x* BB (C) | always |
  | Bet *x* BB / Raise +*x* BB (R) | while raising is allowed; it shows the chips the action adds |
  | Next hand (N or Space) | after a hand ends |

  Limit hold'em has fixed bet sizes: 1 BB on pre-flop and flop, 2 BB on turn
  and river, and at most 4 raises per street.
* At showdown both hands are shown with their names (for example "two pair").
* **Options:**
  * **Show agent's thinking**: shows the agent's action probabilities after
    each of its decisions. They give away information, so this is an inspection mode.
  * **Reveal its cards after a fold**: shows the agent's hand even when the hand ended in a fold.
  * **Seed**: with an integer, a new session with the same seed deals the same sequence of hands; blank deals randomly.
  * **Delay (ms)**: pause before the agent acts.

### Session, logs and review

* The **Session** box shows hands played, your total in BB, and your win rate
  with its uncertainty. A few hundred hands still leaves roughly ±100 mbb/h,
  so treat it as an impression rather than a result.
* Every hand is **auto-saved** to
  `experiments/human_sessions/<opponent>_<date>_<time>.jsonl`.
  **Save session as...** saves an extra copy elsewhere.
* **Review / replay...** opens the hand browser (the same one as the tracker's
  Hands tab) on the current session, or on any hand log via **Open hand log...**.
* Your own statistics: `python behaviour_tracker.py stats <session file> --role you`,
  or load the session file in the tracker GUI.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "No module named ..." in the output, or Check shows MISS | the Interpreter field points at a Python without the packages; select the right one |
| A trained agent won't load | check the kind matches the checkpoint (a PPO checkpoint needs kind `ppo`), and point at the file or its run folder |
| A run overwrote an older run | set an explicit `--log_dir` / output folder; several scripts derive the folder name from the hyperparameters |
| The tuner warns about memory | lower the replay / reservoir sizes in the search space, or run fewer workers |
| Trials stay "running" after a crash | press Start again: the first worker marks interrupted trials as failed |
| The window doesn't fit the screen | the windows can be resized; the action buttons and status bar stay visible |

## Files these GUIs write

| Where | What |
|---|---|
| `<log_dir>/` (training) | `panel.csv/png`, `performance.csv`, `log.txt`, checkpoints |
| `gui_presets/` | training-launcher presets |
| `experiments/tune/<study>/` | `config.json`, `space.json`, `journal.log` (the Optuna study), `trials/`, `finalists/`, `plots/` |
| `experiments/behaviour/<run>/` | `hands.jsonl`, `stats_agent.json`, `stats_opponent.json`, `replays.html` |
| `experiments/human_sessions/` | your play sessions (`*.jsonl`) |
