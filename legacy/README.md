# Legacy files

Kept for reference; not part of the thesis pipeline. They may not run with the
current code (several were written against older interfaces).

| File / folder | What it was | Superseded by |
|---|---|---|
| `limit_* copy.py` | duplicated experiment copies of the training scripts | the scripts in the repo root |
| `limit_dqn_selfplay_vanilla.py` | DQN self-play with rlcard's stock (Tanh) DQN agent | `limit_dqn_selfplay_custom.py` |
| `limit_cfr_selfplay_paper_hyperparams.py` | tabular CFR on limit hold'em (infeasible at that size) | `limit_deepcfr_train_custom.py` |
| `deep_cfr_agent.py` | old TensorFlow-1 Deep CFR | `deep_cfr_agent_new.py` |
| `deep_cfr_load.py` | Deep CFR checkpoint loading | `agent_loading.py` |
| `DQN_hyperparameter_tuner.py` | grid search for DQN | `tune.py` / `tune_gui.py` |
| `head_to_head_template.py`, `head_to_head_template/` | DQN-only head-to-head | `final_evaluation.py` |
| `leduc_dqn_seflplay.py`, `run_cfr.py` | early Leduc hold'em experiments | -- |
| `unlimited.py`, `gpu_check.py`, `*.ipynb` | scratch | -- |
| `experiments_before_fixes/` | results of runs made **before** the bug fixes of October 2026 | new runs under `experiments/` |
| `exploitability_results_before_fixes/` | DQN-responder exploitability results from the same period | `exploitability.py` (LBR) |

**Do not use the results in `experiments_before_fixes/` or
`exploitability_results_before_fixes/`.** They were produced with:
- halved mbb/hand values;
- the HonestAgent with inverted actions;
- a Deep CFR that was not CFR (repeated deals, wiped networks, an invalid fallback strategy, FIFO memories);
- stale raise counts in RLCard's observations;
- the DQN activation option silently ignored.
