# Held-out comparison of finalists

mbb/hand; mean over seeds (sd = spread over seeds). Held-out panel and LBR use deals never used for tuning or for picking the winner; fold rates from duplicate deals vs TAG ("behind": equity below the pot odds, "well ahead": 15 points above).

| study | trial | seeds | tuning score | finalists panel min (tuning deals) | held-out panel min | LBR (lower = better) | vs station | vs maniac | vs nit | vs tag | vs lag | fold facing bet | fold when behind | fold when well ahead | VPIP | PFR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_v1 | 37 | 3 | -416 | -449 (sd 13) | -514 (sd 11) | +732 (sd 16) | +43 (sd 8) | +199 (sd 101) | -514 (sd 11) | -374 (sd 8) | -264 (sd 15) | 0% (sd 0%) | 0% (sd 0%) | 0% (sd 0%) | 91% (sd 1%) | 2% (sd 1%) |
| ppo_v1 | 26 (winner) | 3 | -472 | -389 (sd 21) | -432 (sd 31) | +821 (sd 43) | +315 (sd 49) | +298 (sd 43) | -409 (sd 24) | -432 (sd 31) | -336 (sd 17) | 0% (sd 0%) | 0% (sd 0%) | 0% (sd 0%) | 93% (sd 0%) | 18% (sd 3%) |
| ppo_v1 | 15 | 3 | -485 | -390 (sd 19) | -441 (sd 12) | +851 (sd 56) | +298 (sd 53) | +251 (sd 43) | -441 (sd 12) | -423 (sd 21) | -329 (sd 14) | 0% (sd 0%) | 0% (sd 0%) | 0% (sd 0%) | 92% (sd 1%) | 12% (sd 3%) |
| ppo_v2 | 35 (winner) | 3 | -171 | -367 (sd 31) | -319 (sd 22) | +813 (sd 132) | +98 (sd 47) | -261 (sd 34) | -198 (sd 62) | -305 (sd 46) | -313 (sd 16) | 20% (sd 5%) | 25% (sd 13%) | 14% (sd 4%) | 76% (sd 8%) | 45% (sd 13%) |
| ppo_v2 | 0 | 3 | -330 | -676 (sd 316) | -700 (sd 360) | +838 (sd 394) | -89 (sd 95) | -570 (sd 484) | -261 (sd 64) | -491 (sd 66) | -528 (sd 139) | 33% (sd 18%) | 23% (sd 29%) | 26% (sd 16%) | 60% (sd 20%) | 16% (sd 7%) |
| ppo_v2 | 32 | 3 | -354 | -1077 (sd 45) | -1044 (sd 109) | +1258 (sd 48) | +89 (sd 64) | +437 (sd 250) | -609 (sd 40) | -997 (sd 102) | -1044 (sd 109) | 16% (sd 5%) | 18% (sd 3%) | 14% (sd 5%) | 78% (sd 5%) | 54% (sd 4%) |
