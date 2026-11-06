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
| dqn_v1 | 0 | 3 | -673 | -1385 (sd 735) | -1373 (sd 796) | +2234 (sd 1504) | +93 (sd 213) | -1157 (sd 951) | -662 (sd 586) | -1155 (sd 724) | -1345 (sd 750) | 7% (sd 6%) | 41% (sd 28%) | 5% (sd 4%) | 93% (sd 4%) | 71% (sd 27%) |
| dqn_v1 | 19 | 3 | -783 | -1111 (sd 69) | -1083 (sd 110) | +1521 (sd 66) | +158 (sd 16) | -503 (sd 214) | -494 (sd 11) | -991 (sd 58) | -1083 (sd 110) | 11% (sd 0%) | 63% (sd 8%) | 7% (sd 1%) | 92% (sd 0%) | 68% (sd 2%) |
| dqn_v1 | 7 (winner) | 3 | -1282 | -1072 (sd 66) | -1126 (sd 145) | +1286 (sd 117) | +250 (sd 26) | -1126 (sd 145) | -433 (sd 33) | -776 (sd 68) | -873 (sd 114) | 16% (sd 1%) | 82% (sd 3%) | 10% (sd 0%) | 86% (sd 1%) | 56% (sd 4%) |
| deepcfr_v1 | 24 | 3 | -346 | -491 (sd 6) | -457 (sd 30) | +752 (sd 51) | +340 (sd 29) | +306 (sd 64) | -325 (sd 10) | -431 (sd 3) | -457 (sd 30) | 22% (sd 1%) | 18% (sd 1%) | 18% (sd 1%) | 69% (sd 2%) | 27% (sd 1%) |
| deepcfr_v1 | 27 (winner) | 3 | -379 | -439 (sd 39) | -459 (sd 43) | +620 (sd 27) | +254 (sd 30) | +455 (sd 99) | -362 (sd 27) | -451 (sd 33) | -449 (sd 55) | 21% (sd 1%) | 23% (sd 3%) | 16% (sd 1%) | 70% (sd 2%) | 24% (sd 3%) |
| deepcfr_v1 | 0 | 3 | -392 | -505 (sd 60) | -477 (sd 63) | +563 (sd 112) | +326 (sd 20) | +748 (sd 104) | -337 (sd 53) | -477 (sd 63) | -434 (sd 76) | 20% (sd 3%) | 15% (sd 4%) | 15% (sd 2%) | 68% (sd 4%) | 33% (sd 3%) |
| nfsp_v2 | 40 (winner) | 3 | -438 | -366 (sd 10) | -460 (sd 13) | +682 (sd 13) | -41 (sd 16) | -185 (sd 174) | -460 (sd 13) | -414 (sd 26) | -343 (sd 21) | 1% (sd 0%) | 0% (sd 1%) | 1% (sd 0%) | 98% (sd 1%) | 11% (sd 0%) |
| nfsp_v2 | 36 | 2 | -441 | -361 (sd 1) | -475 (sd 6) | +612 (sd 13) | -19 (sd 6) | -28 (sd 5) | -475 (sd 6) | -399 (sd 30) | -318 (sd 11) | 0% (sd 0%) | 0% (sd 0%) | 0% (sd 0%) | 98% (sd 0%) | 11% (sd 1%) |
| nfsp_v2 | 18 | 3 | -446 | -367 (sd 7) | -476 (sd 6) | +675 (sd 51) | -27 (sd 13) | -39 (sd 6) | -476 (sd 6) | -407 (sd 11) | -316 (sd 17) | 0% (sd 0%) | 0% (sd 0%) | 0% (sd 0%) | 98% (sd 0%) | 11% (sd 1%) |
