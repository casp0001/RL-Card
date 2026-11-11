Leduc hold'em, exact evaluation of the finalists (top 3 trials x 3 seeds of each study).
Reference: tabular CFR+ (1000 iterations) has exploitability 0.09 mbb/hand, the uniform random policy 1472.

| algorithm | study | trial | seeds | exploitability (mbb/hand, lower = better) | vs equilibrium | vs random | winner |
|---|---|---|---|---|---|---|---|
| deepcfr | leduc_deepcfr_v1 | 8 | 3 | 187 (sd 17) | -31 (sd 2) | 701 (sd 3) |  |
| deepcfr | leduc_deepcfr_v1 | 29 | 3 | 179 (sd 14) | -28 (sd 3) | 712 (sd 17) |  |
| deepcfr | leduc_deepcfr_v1 | 15 | 3 | 175 (sd 10) | -27 (sd 3) | 710 (sd 8) | yes |
| ppo | leduc_ppo_v1 | 1 | 3 | 962 (sd 396) | -158 (sd 121) | 842 (sd 116) |  |
| ppo | leduc_ppo_v1 | 27 | 3 | 813 (sd 183) | -165 (sd 89) | 837 (sd 66) | yes |
| ppo | leduc_ppo_v1 | 25 | 3 | 1233 (sd 31) | -47 (sd 15) | 632 (sd 29) |  |
| nfsp | leduc_nfsp_v1 | 9 | 3 | 343 (sd 9) | -115 (sd 18) | 913 (sd 2) |  |
| nfsp | leduc_nfsp_v1 | 27 | 3 | 315 (sd 16) | -145 (sd 8) | 943 (sd 18) | yes |
| nfsp | leduc_nfsp_v1 | 24 | 3 | 381 (sd 35) | -97 (sd 6) | 899 (sd 12) |  |
| dqn | leduc_dqn_v1 | 22 | 3 | 1011 (sd 69) | -116 (sd 74) | 1066 (sd 130) | yes |
| dqn | leduc_dqn_v1 | 29 | 3 | 1156 (sd 371) | -92 (sd 58) | 1041 (sd 119) |  |
| dqn | leduc_dqn_v1 | 18 | 3 | 1122 (sd 155) | -136 (sd 36) | 836 (sd 113) |  |
