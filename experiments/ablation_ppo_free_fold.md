### Free-fold ablation, PPO (ppo_v2 trial 35), mean (sd) over seeds [1042, 1043, 1044]

| arm | P(fold | check free) | panel min | panel mean | vs random | vs station | vs maniac | vs nit | vs tag | vs lag | LBR | VPIP | PFR | fold facing a bet | AF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: original (free fold allowed) | 5.9% (sd 2.1) | -319 (sd 22) | -196 (sd 30) | +1079 (sd 175) | +98 (sd 47) | -261 (sd 34) | -198 (sd 62) | -305 (sd 46) | -313 (sd 16) | +813 (sd 132) | 76% (sd 8) | 45% (sd 13) | 20% (sd 5) | 2.95 (sd 0.98) |
| B: A, fold moved to check at evaluation | 0.0% (sd 0.0) | -269 (sd 13) | -108 (sd 31) | +1161 (sd 131) | +278 (sd 80) | -259 (sd 30) | -146 (sd 61) | -206 (sd 42) | -208 (sd 11) | +778 (sd 126) | 76% (sd 8) | 45% (sd 13) | 21% (sd 6) | 2.91 (sd 0.96) |
| C: retrained without free fold | 0.0% (sd 0.0) | -341 (sd 323) | -75 (sd 83) | +1446 (sd 70) | +298 (sd 40) | -283 (sd 374) | -146 (sd 28) | -129 (sd 15) | -117 (sd 44) | +725 (sd 183) | 89% (sd 3) | 47% (sd 1) | 13% (sd 2) | 3.46 (sd 0.20) |

Held-out panel (2000 duplicate deals per bot, FINAL_PANEL_SEED), LBR (1000 deals, lower = less exploitable), style from 500 deals against tag. mbb/hand.

### Paired differences per seed (mean (sd) over seeds)

| difference | panel min | panel mean | LBR (negative = less exploitable) |
|---|---|---|---|
| C - A | -22 (sd 345) | +121 (sd 111) | -88 (sd 309) |
| C - B | -72 (sd 316) | +33 (sd 112) | -53 (sd 304) |
| B - A | +51 (sd 30) | +88 (sd 25) | -35 (sd 9) |

### Head-to-head (2000 duplicate deals per pair, mean over seed pairs)

| match | row wins (mbb/hand) |
|---|---|
| C vs A | +133 ± 54 |
| C vs B | +82 ± 49 |
| B vs A | +48 ± 40 |
| A vs deepcfr | +65 ± 42 |
| B vs deepcfr | +114 ± 31 |
| C vs deepcfr | +178 ± 44 |

± counts the training seeds of both sides as the random units plus the card noise.

### Training-time panel score (mean of the last 3 evaluations, tuning deals)

| arm | score | training hours |
|---|---|---|
| A: original | -285 (sd 39) |  |
| C: retrained | -319 (sd 117) | 0.6 (sd 0.0) |
