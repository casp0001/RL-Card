# Hyper-parameter tuning

### DQN, limit: study dqn_v1

| parameter | searched | winner | importance (5 complete) | importance (50 trials at evaluation 2) |
|---|---|---|---|---|
| lr | 1e-05 - 0.003 (log) | 1.03e-05 | 0.31 | 0.07 |
| batch_size | 32 / 64 / 128 / 256 / 512 | 512 | 0.11 | 0.16 |
| mlp_layers | 128 128 / 256 256 / 512 512 / 256 256 256 / 512 512 512 | 256 256 | 0.17 | 0.23 |
| epsilon_decay_steps | 50000 - 3e+06 (log) | 321427 | 0.07 | 0.10 |
| epsilon_end | 0.01 - 0.25 (log) | 0.0203 | 0.02 | 0.31 |
| replay_memory_size | 20000 - 1e+06 (log) | 331814 | 0.08 | 0.04 |
| opponent_update_every | 1000 - 100000 (log) | 4186 | 0.05 | 0.02 |
| update_target_every | 250 - 10000 (log) | 666 | 0.18 | 0.06 |
| train_every | fixed 1 | (fixed) |  |  |
| activation | fixed relu | (fixed) |  |  |

50 trials (complete 5, pruned 45); budget 1,000,000 episodes, 10 evaluations per trial, successive-halving pruning; winner = trial 7, chosen by `tune.py finalists` (top 3 trials x 3 new seeds). Fixed settings: obs=rich, replay_memory_init_size=5000, discount_factor=1.0, warmup_episodes=200000, warmup_bot=honest. Importance: fANOVA share of the score variance explained by each parameter, over the complete trials, and over every trial's score at evaluation 2, the first one at which pruning could stop a trial (so all trials reached it); blank = too few trials.

### PPO, limit: study ppo_v2

| parameter | searched | winner | importance (7 complete) | importance (50 trials at evaluation 2) |
|---|---|---|---|---|
| lr | 5e-05 - 0.001 (log) | 0.000462 | 0.16 | 0.20 |
| update_every | 2048 / 4096 / 8192 / 16384 | 4096 | 0.00 | 0.48 |
| num_epochs | 2 - 8 | 4 | 0.16 | 0.10 |
| batch_size | 128 / 256 / 512 / 1024 | 1024 | 0.16 | 0.02 |
| clip_range | 0.1 - 0.3 | 0.186 | 0.08 | 0.03 |
| entropy_coef | 0.001 - 0.03 (log) | 0.0169 | 0.19 | 0.13 |
| gae_lambda | fixed 0.95 | (fixed) |  |  |
| mlp_layers | fixed 256 256 | (fixed) |  |  |
| opponent_update_every | 2000 - 50000 (log) | 9321 | 0.26 | 0.04 |

50 trials (complete 7, pruned 43); budget 1,000,000 episodes, 10 evaluations per trial, successive-halving pruning; winner = trial 35, chosen by `tune.py finalists` (top 3 trials x 3 new seeds). Fixed settings: obs=rich, discount_factor=1.0, warmup_episodes=200000, warmup_bot=honest. Importance: fANOVA share of the score variance explained by each parameter, over the complete trials, and over every trial's score at evaluation 2, the first one at which pruning could stop a trial (so all trials reached it); blank = too few trials.

### NFSP, limit: study nfsp_v3

| parameter | searched | winner | importance (9 complete) | importance (50 trials at evaluation 2) |
|---|---|---|---|---|
| rl_learning_rate | 5e-06 - 0.001 (log) | 5.4e-06 | 0.00 | 0.00 |
| sl_learning_rate | 1e-05 - 0.003 (log) | 0.00165 | 0.08 | 0.04 |
| anticipatory_param | 0.03 - 0.9 (log) | 0.0327 | 0.04 | 0.01 |
| reservoir_buffer_capacity | 200000 - 1.5e+06 (log) | 354192 | 0.10 | 0.02 |
| q_replay_memory_size | 20000 - 600000 (log) | 78922 | 0.13 | 0.01 |
| q_epsilon_start | 0.02 - 0.3 (log) | 0.0214 | 0.50 | 0.82 |
| layers | 128 128 / 256 256 / 512 512 / 256 256 256 | 256 256 | 0.05 | 0.01 |
| train_every | 16 / 32 / 64 / 128 | 16 | 0.01 | 0.01 |
| optimizer | fixed adam | (fixed) |  |  |
| q_activation | fixed relu | (fixed) |  |  |
| opponent_update_every | 1000 - 100000 (log) | 18149 | 0.09 | 0.08 |
| max_grad_norm | fixed 10.0 | (fixed) |  |  |

50 trials (complete 9, pruned 41); budget 2,000,000 episodes, 10 evaluations per trial, successive-halving pruning; winner = trial 33, chosen by `tune.py finalists` (top 3 trials x 3 new seeds). Fixed settings: obs=rich, q_discount_factor=1.0, eval_best_response=False, q_replay_memory_init_size=5000, min_buffer_size_to_learn=5000, warmup_episodes=400000, warmup_bot=honest. Importance: fANOVA share of the score variance explained by each parameter, over the complete trials, and over every trial's score at evaluation 2, the first one at which pruning could stop a trial (so all trials reached it); blank = too few trials.

### Deep CFR, limit: study deepcfr_v1

| parameter | searched | winner | importance (4 complete) | importance (30 trials at evaluation 2) |
|---|---|---|---|---|
| lr | 0.0001 - 0.003 (log) | 0.000653 |  | 0.67 |
| num_traversals | fixed 1000 | (fixed) |  |  |
| num_step | fixed 1000 | (fixed) |  |  |
| batch_size | 1024 / 4096 / 8192 | 8192 |  | 0.28 |
| layers | 128 128 128 / 256 256 256 | 128 128 128 |  | 0.04 |
| memory_capacity | fixed 2000000 | (fixed) |  |  |

30 trials (complete 4, pruned 26); budget 200 iterations, 10 evaluations per trial, successive-halving pruning; winner = trial 27, chosen by `tune.py finalists` (top 3 trials x 3 new seeds). Fixed settings: obs=rich. Importance: fANOVA share of the score variance explained by each parameter, over the complete trials, and over every trial's score at evaluation 2, the first one at which pruning could stop a trial (so all trials reached it); blank = too few trials.

### DQN, leduc: study leduc_dqn_v1

| parameter | searched | winner | importance (9 complete) | importance (30 trials at evaluation 5) |
|---|---|---|---|---|
| lr | 1e-05 - 0.003 (log) | 0.000188 | 0.04 | 0.07 |
| batch_size | 32 / 64 / 128 / 256 / 512 | 32 | 0.12 | 0.08 |
| mlp_layers | 128 128 / 256 256 / 512 512 / 256 256 256 / 512 512 512 | 256 256 | 0.08 | 0.09 |
| epsilon_decay_steps | 50000 - 3e+06 (log) | 94232 | 0.09 | 0.41 |
| epsilon_end | 0.01 - 0.25 (log) | 0.246 | 0.10 | 0.13 |
| replay_memory_size | 20000 - 1e+06 (log) | 170545 | 0.07 | 0.05 |
| opponent_update_every | 1000 - 100000 (log) | 2224 | 0.25 | 0.10 |
| update_target_every | 250 - 10000 (log) | 477 | 0.24 | 0.06 |
| train_every | fixed 1 | (fixed) |  |  |
| activation | fixed relu | (fixed) |  |  |

30 trials (complete 9, pruned 21); budget 1,000,000 episodes, 20 evaluations per trial, successive-halving pruning; winner = trial 22, chosen by `tune.py finalists` (top 3 trials x 3 new seeds). Fixed settings: obs=rich, replay_memory_init_size=5000, discount_factor=1.0, game=leduc. Importance: fANOVA share of the score variance explained by each parameter, over the complete trials, and over every trial's score at evaluation 5, the first one at which pruning could stop a trial (so all trials reached it); blank = too few trials.

### PPO, leduc: study leduc_ppo_v1

| parameter | searched | winner | importance (5 complete) | importance (30 trials at evaluation 5) |
|---|---|---|---|---|
| lr | 5e-05 - 0.001 (log) | 7.45e-05 | 0.11 | 0.12 |
| update_every | 2048 / 4096 / 8192 / 16384 | 8192 | 0.09 | 0.03 |
| num_epochs | 2 - 8 | 3 | 0.31 | 0.35 |
| batch_size | 128 / 256 / 512 / 1024 | 1024 | 0.08 | 0.30 |
| clip_range | 0.1 - 0.3 | 0.163 | 0.16 | 0.08 |
| entropy_coef | 0.001 - 0.03 (log) | 0.0064 | 0.05 | 0.07 |
| gae_lambda | fixed 0.95 | (fixed) |  |  |
| mlp_layers | fixed 256 256 | (fixed) |  |  |
| opponent_update_every | 2000 - 50000 (log) | 12186 | 0.20 | 0.05 |

30 trials (complete 5, pruned 25); budget 1,000,000 episodes, 20 evaluations per trial, successive-halving pruning; winner = trial 27, chosen by `tune.py finalists` (top 3 trials x 3 new seeds). Fixed settings: obs=rich, discount_factor=1.0, game=leduc. Importance: fANOVA share of the score variance explained by each parameter, over the complete trials, and over every trial's score at evaluation 5, the first one at which pruning could stop a trial (so all trials reached it); blank = too few trials.

### NFSP, leduc: study leduc_nfsp_v1

| parameter | searched | winner | importance (7 complete) | importance (30 trials at evaluation 5) |
|---|---|---|---|---|
| rl_learning_rate | 5e-06 - 0.001 (log) | 0.000252 | 0.20 | 0.10 |
| sl_learning_rate | 1e-05 - 0.003 (log) | 1.44e-05 | 0.04 | 0.00 |
| anticipatory_param | 0.03 - 0.9 (log) | 0.0387 | 0.10 | 0.03 |
| reservoir_buffer_capacity | 200000 - 1.5e+06 (log) | 515097 | 0.09 | 0.03 |
| q_replay_memory_size | 20000 - 600000 (log) | 25893 | 0.26 | 0.01 |
| q_epsilon_start | 0.02 - 0.3 (log) | 0.0429 | 0.13 | 0.72 |
| layers | 128 128 / 256 256 / 512 512 / 256 256 256 | 128 128 | 0.11 | 0.04 |
| train_every | 16 / 32 / 64 / 128 | 32 | 0.01 | 0.04 |
| optimizer | fixed adam | (fixed) |  |  |
| q_activation | fixed relu | (fixed) |  |  |
| opponent_update_every | 1000 - 100000 (log) | 34927 | 0.05 | 0.03 |
| max_grad_norm | fixed 10.0 | (fixed) |  |  |

30 trials (complete 7, pruned 23); budget 2,000,000 episodes, 20 evaluations per trial, successive-halving pruning; winner = trial 27, chosen by `tune.py finalists` (top 3 trials x 3 new seeds). Fixed settings: obs=rich, q_discount_factor=1.0, eval_best_response=False, q_replay_memory_init_size=5000, min_buffer_size_to_learn=5000, game=leduc. Importance: fANOVA share of the score variance explained by each parameter, over the complete trials, and over every trial's score at evaluation 5, the first one at which pruning could stop a trial (so all trials reached it); blank = too few trials.

### Deep CFR, leduc: study leduc_deepcfr_v1

| parameter | searched | winner | importance (7 complete) | importance (30 trials at evaluation 5) |
|---|---|---|---|---|
| lr | 0.0001 - 0.003 (log) | 0.00294 | 0.60 | 0.88 |
| num_traversals | fixed 500 | (fixed) |  |  |
| num_step | fixed 300 | (fixed) |  |  |
| batch_size | 512 / 1024 / 2048 | 1024 | 0.18 | 0.04 |
| layers | 64 64 / 128 128 / 128 128 128 / 256 256 | 128 128 | 0.22 | 0.08 |
| memory_capacity | fixed 1000000 | (fixed) |  |  |

30 trials (complete 7, pruned 23); budget 300 iterations, 20 evaluations per trial, successive-halving pruning; winner = trial 15, chosen by `tune.py finalists` (top 3 trials x 3 new seeds). Fixed settings: obs=rich, game=leduc. Importance: fANOVA share of the score variance explained by each parameter, over the complete trials, and over every trial's score at evaluation 5, the first one at which pruning could stop a trial (so all trials reached it); blank = too few trials.
