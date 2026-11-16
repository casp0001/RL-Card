### BatchNorm ablation, NFSP (nfsp_v3 trial 33), mean (sd) over seeds [1042, 1043, 1044]

| arm | P(fold | check free) | panel min | panel mean | vs random | vs station | vs maniac | vs nit | vs tag | vs lag | LBR | VPIP | PFR | fold facing a bet | AF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: original (BatchNorm on the input of both networks) | 1.8% (sd 0.2) | -1107 (sd 48) | -419 (sd 38) | +1528 (sd 74) | +113 (sd 25) | +655 (sd 140) | -685 (sd 31) | -1071 (sd 37) | -1107 (sd 48) | +1611 (sd 136) | 73% (sd 1) | 52% (sd 3) | 14% (sd 1) | 3.87 (sd 0.66) |
| C: retrained without BatchNorm | 3.5% (sd 0.5) | -1357 (sd 85) | -522 (sd 67) | +1319 (sd 51) | +8 (sd 39) | +754 (sd 140) | -788 (sd 83) | -1228 (sd 99) | -1357 (sd 85) | +1906 (sd 127) | 65% (sd 1) | 43% (sd 7) | 20% (sd 1) | 7.15 (sd 1.73) |

Held-out panel (2000 duplicate deals per bot, FINAL_PANEL_SEED), LBR (1000 deals, lower = less exploitable), style from 500 deals against tag. mbb/hand.

### Paired differences per seed (mean (sd) over seeds)

| difference | panel min | panel mean | LBR (negative = less exploitable) |
|---|---|---|---|
| C - A | -250 (sd 91) | -103 (sd 35) | +296 (sd 179) |

### Head-to-head (2000 duplicate deals per pair, mean over seed pairs)

| match | row wins (mbb/hand) |
|---|---|
| C vs A | -22 ± 59 |
| A vs deepcfr | -167 ± 49 |
| C vs deepcfr | -230 ± 42 |

± counts the training seeds of both sides as the random units plus the card noise.

### Training-time panel score (mean of the last 3 evaluations, tuning deals)

| arm | score | training hours |
|---|---|---|
| A: original | -1084 (sd 117) |  |
| C: retrained | -1409 (sd 119) | 2.4 (sd 0.1) |
