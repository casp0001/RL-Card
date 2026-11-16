### NFSP tuning studies: diverged best-response Q-networks (v1, v2) and the fixed study (v3)

| study | optimizer | rl learning rate searched | gradient clipping | trials (complete / pruned / failed) | best tuning score | finalists with a non-finite Q-network | held-out panel min (winner) | LBR (winner) | fold facing a bet vs tag (winner) |
|---|---|---|---|---|---|---|---|---|---|
| nfsp_v1 | sgd | 0.002 - 0.3 (log) | none | 39 (11 / 24 / 0) | -492 (trial 27, rl lr 0.17) | no finalists |  |  |  |
| nfsp_v2 | sgd | 0.02 - 1 (log) | none | 50 (9 / 41 / 0) | -438 (trial 40, rl lr 0.34) | 8 / 8 | -460 (sd 13) | 682 (sd 13) | 1% |
| nfsp_v3 | adam | 5e-06 - 0.001 (log) | fixed 10.0 | 50 (9 / 41 / 0) | -1127 (trial 44, rl lr 8.69e-06) | 0 / 9 | -1107 (sd 48) | 1611 (sd 136) | 14% |

In nfsp_v1 / nfsp_v2 (plain SGD, large RL learning rates, no divergence check) the best-response Q-network diverged to NaN; the greedy action then silently became the first legal one (call, else raise), so the average policy learned never to fold, which the panel rewarded and the tuner selected. nfsp_v3 uses Adam, smaller learning rates, gradient clipping and a divergence check that fails such trials. Held-out values: mean (sd) over the winner's seeds; fold rate from the held-out match against tag.
