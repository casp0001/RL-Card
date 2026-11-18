### SGD-steps ablation, Deep CFR (deepcfr_v1 trial 27), mean (sd) over seeds [1042, 1043, 1044]

| arm | P(fold | check free) | panel min | panel mean | vs random | vs station | vs maniac | vs nit | vs tag | vs lag | LBR | VPIP | PFR | fold facing a bet | AF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: original (1000 SGD steps per network fit) | 1.0% (sd 0.3) | -459 (sd 43) | -111 (sd 46) | +1016 (sd 39) | +254 (sd 30) | +455 (sd 99) | -362 (sd 27) | -451 (sd 33) | -449 (sd 55) | +620 (sd 27) | 70% (sd 2) | 24% (sd 3) | 21% (sd 1) | 2.48 (sd 0.07) |
| C: retrained with 4000 SGD steps per fit | 2.8% (sd 0.1) | -612 (sd 105) | -293 (sd 60) | +987 (sd 43) | +132 (sd 13) | -44 (sd 109) | -372 (sd 21) | -571 (sd 66) | -612 (sd 105) | +991 (sd 42) | 69% (sd 1) | 46% (sd 1) | 24% (sd 0) | 3.79 (sd 0.26) |
| reference (ppo_ext) | 2.6% (sd 1.2) | -270 (sd 49) | -64 (sd 5) | +1222 (sd 149) | +233 (sd 62) | +68 (sd 170) | -191 (sd 50) | -270 (sd 49) | -163 (sd 17) | +627 (sd 199) | 77% (sd 5) | 45% (sd 9) | 18% (sd 3) | 2.89 (sd 0.66) |
| reference (deepcfr_ext) | 0.8% (sd 0.2) | -331 (sd 13) | -43 (sd 31) | +1205 (sd 54) | +324 (sd 19) | +359 (sd 118) | -266 (sd 9) | -328 (sd 14) | -306 (sd 40) | +597 (sd 118) | 70% (sd 2) | 42% (sd 2) | 21% (sd 1) | 3.70 (sd 0.15) |

Held-out panel (2000 duplicate deals per bot, FINAL_PANEL_SEED), LBR (1000 deals, lower = less exploitable), style from 500 deals against tag. mbb/hand.

### Paired differences per seed (mean (sd) over seeds)

| difference | panel min | panel mean | LBR (negative = less exploitable) |
|---|---|---|---|
| C - A | -153 (sd 105) | -183 (sd 77) | +371 (sd 67) |

### Head-to-head (2000 duplicate deals per pair, mean over seed pairs)

| match | row wins (mbb/hand) |
|---|---|
| C vs A | -120 ± 46 |
| A vs ppo_ext | -189 ± 21 |
| C vs ppo_ext | -296 ± 28 |
| A vs deepcfr_ext | -128 ± 25 |
| C vs deepcfr_ext | -217 ± 34 |

± counts the training seeds of both sides as the random units plus the card noise.

### Training-time panel score (mean of the last 3 evaluations, tuning deals)

| arm | score | training hours |
|---|---|---|
| A: original (1000 SGD steps per network fit) | -373 (sd 26) |  |
| C: retrained | -675 (sd 76) | 35.0 (sd 0.2) |
