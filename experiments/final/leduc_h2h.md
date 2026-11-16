# Leduc hold'em: exact head-to-head

Every winner seed against every seed of the other algorithms, computed exactly over all deals (no card noise). mbb/hand from the row's view, averaged over both seats.

### Leduc, exact head-to-head, every agent

| agent | dqn_s1042 | dqn_s1043 | dqn_s1044 | ppo_s1042 | ppo_s1043 | ppo_s1044 | nfsp_s1042 | nfsp_s1043 | nfsp_s1044 | deepcfr_s1042 | deepcfr_s1043 | deepcfr_s1044 | CFR+ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dqn_s1042 | -- | -- | -- | -113 | +230 | -266 | +81 | +79 | +66 | -70 | -110 | -101 | -124 |
| dqn_s1043 | -- | -- | -- | +193 | +136 | +38 | +22 | +50 | +45 | -123 | -115 | -89 | -186 |
| dqn_s1044 | -- | -- | -- | +388 | +343 | +368 | +107 | +74 | +108 | -37 | -18 | -39 | -39 |
| ppo_s1042 | +113 | -193 | -388 | -- | -- | -- | -137 | -143 | -151 | -254 | -255 | -240 | -267 |
| ppo_s1043 | -230 | -136 | -343 | -- | -- | -- | -72 | -53 | -82 | -95 | -85 | -84 | -112 |
| ppo_s1044 | +266 | -38 | -368 | -- | -- | -- | -30 | -34 | -57 | -112 | -94 | -104 | -115 |
| nfsp_s1042 | -81 | -22 | -107 | +137 | +72 | +30 | -- | -- | -- | -67 | -72 | -67 | -150 |
| nfsp_s1043 | -79 | -50 | -74 | +143 | +53 | +34 | -- | -- | -- | -65 | -69 | -61 | -149 |
| nfsp_s1044 | -66 | -45 | -108 | +151 | +82 | +57 | -- | -- | -- | -63 | -64 | -58 | -136 |
| deepcfr_s1042 | +70 | +123 | +37 | +254 | +95 | +112 | +67 | +65 | +63 | -- | -- | -- | -28 |
| deepcfr_s1043 | +110 | +115 | +18 | +255 | +85 | +94 | +72 | +69 | +64 | -- | -- | -- | -24 |
| deepcfr_s1044 | +101 | +89 | +39 | +240 | +84 | +104 | +67 | +61 | +58 | -- | -- | -- | -29 |

### Leduc, exact head-to-head by algorithm (mean over seed pairs, sd = spread)

| algorithm | DQN | PPO | NFSP | Deep CFR | CFR+ equilibrium |
|---|---|---|---|---|---|
| DQN | -- | +146 (sd 225) | +70 (sd 28) | -78 (sd 39) | -116 (sd 74) |
| PPO | -146 (sd 225) | -- | -84 (sd 48) | -147 (sd 78) | -165 (sd 89) |
| NFSP | -70 (sd 28) | +84 (sd 48) | -- | -65 (sd 4) | -145 (sd 8) |
| Deep CFR | +78 (sd 39) | +147 (sd 78) | +65 (sd 4) | -- | -27 (sd 3) |

### Leduc: ranking by head-to-head and by exploitability

| algorithm | mean head-to-head vs others (mbb/hand) | exploitability (mbb/hand) |
|---|---|---|
| Deep CFR | +97 | 175 |
| DQN | +46 | 1011 |
| NFSP | -17 | 315 |
| PPO | -126 | 813 |

Spearman rank correlation over the 12 agents between the mean head-to-head result and exploitability (negated, so +1 = same order): +0.31.
