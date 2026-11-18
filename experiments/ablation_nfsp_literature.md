### Literature-configuration check, NFSP (8M episodes) (nfsp_v3 trial 33), mean (sd) over seeds [1042, 1043, 1044]

| arm | P(fold | check free) | panel min | panel mean | vs random | vs station | vs maniac | vs nit | vs tag | vs lag | LBR | VPIP | PFR | fold facing a bet | AF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: tuned (nfsp_v3 winner, extended) | 2.5% (sd 0.4) | -1085 (sd 140) | -352 (sd 60) | +1518 (sd 47) | +298 (sd 16) | +678 (sd 163) | -622 (sd 38) | -1029 (sd 94) | -1084 (sd 141) | +1258 (sd 74) | 83% (sd 1) | 37% (sd 2) | 11% (sd 1) | 3.76 (sd 0.65) |
| C: literature values (eta 0.1, eps 0.08, RL lr 1e-4, reservoir 1.5M, replay 200k) | 2.3% (sd 0.3) | -1294 (sd 167) | -361 (sd 70) | +2035 (sd 81) | +295 (sd 55) | +1025 (sd 57) | -658 (sd 47) | -1174 (sd 96) | -1294 (sd 167) | +1582 (sd 134) | 96% (sd 0) | 68% (sd 2) | 5% (sd 1) | 6.13 (sd 1.07) |
| reference (ppo_ext) | 2.6% (sd 1.2) | -270 (sd 49) | -64 (sd 5) | +1222 (sd 149) | +233 (sd 62) | +68 (sd 170) | -191 (sd 50) | -270 (sd 49) | -163 (sd 17) | +627 (sd 199) | 77% (sd 5) | 45% (sd 9) | 18% (sd 3) | 2.89 (sd 0.66) |
| reference (deepcfr_ext) | 0.8% (sd 0.2) | -331 (sd 13) | -43 (sd 31) | +1205 (sd 54) | +324 (sd 19) | +359 (sd 118) | -266 (sd 9) | -328 (sd 14) | -306 (sd 40) | +597 (sd 118) | 70% (sd 2) | 42% (sd 2) | 21% (sd 1) | 3.70 (sd 0.15) |

Held-out panel (2000 duplicate deals per bot, FINAL_PANEL_SEED), LBR (1000 deals, lower = less exploitable), style from 500 deals against tag. mbb/hand.

### Paired differences per seed (mean (sd) over seeds)

| difference | panel min | panel mean | LBR (negative = less exploitable) |
|---|---|---|---|
| C - A | -209 (sd 190) | -9 (sd 114) | +324 (sd 161) |

### Head-to-head (2000 duplicate deals per pair, mean over seed pairs)

| match | row wins (mbb/hand) |
|---|---|
| C vs A | +64 ± 71 |
| A vs ppo_ext | -204 ± 44 |
| C vs ppo_ext | -136 ± 68 |
| A vs deepcfr_ext | -140 ± 31 |
| C vs deepcfr_ext | -132 ± 46 |

± counts the training seeds of both sides as the random units plus the card noise.

### Training-time panel score (mean of the last 3 evaluations, tuning deals)

| arm | score | training hours |
|---|---|---|
| A: tuned (nfsp_v3 winner, extended) | -1052 (sd 49) | 8.6 (sd 2.5) |
| C: retrained | -1319 (sd 30) | 24.8 (sd 0.2) |
