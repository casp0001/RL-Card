# Final evaluation

Settings: {'h2h_deals': 2000, 'panel_deals': 5000, 'lbr_deals': 5000, 'lbr_bot_deals': 500, 'lbr_samples': 500, 'panel_seed': 50000000, 'h2h_details': 1, 'style_deals': 500}. All values in mbb/hand; ± is one standard error (per agent) or the standard deviation over seeds (per algorithm, "sd").

### Head-to-head, every agent (row vs column)

| agent | dqn_s1042 | dqn_s1043 | dqn_s1044 | ppo_s1043 | ppo_s1042 | ppo_s1044 | nfsp_s1042 | nfsp_s1043 | nfsp_s1044 | deepcfr_s1042 | deepcfr_s1044 | deepcfr_s1043 | ppo_ext_s1042 | ppo_ext_s1043 | ppo_ext_s1044 | nfsp_ext_s1042 | nfsp_ext_s1043 | nfsp_ext_s1044 | deepcfr_ext_s1042 | deepcfr_ext_s1043 | deepcfr_ext_s1044 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dqn_s1042 | -- |  |  | -246 ± 56 | -65 ± 50 | -92 ± 60 | -100 ± 80 | -123 ± 82 | -63 ± 81 | -178 ± 61 | -155 ± 62 | -210 ± 60 | -383 ± 61 | -296 ± 57 | -270 ± 54 | -53 ± 74 | -131 ± 76 | -102 ± 81 | -312 ± 64 | -206 ± 64 | -286 ± 64 |
| dqn_s1043 |  | -- |  | -70 ± 58 | +3 ± 53 | -52 ± 61 | -147 ± 85 | -241 ± 91 | -183 ± 88 | -53 ± 62 | -116 ± 64 | -143 ± 64 | -303 ± 63 | -154 ± 59 | -316 ± 57 | -268 ± 81 | -234 ± 82 | -303 ± 88 | -273 ± 66 | -196 ± 67 | -249 ± 66 |
| dqn_s1044 |  |  | -- | -124 ± 57 | +26 ± 53 | -126 ± 62 | -116 ± 85 | -283 ± 89 | -84 ± 87 | -142 ± 63 | -116 ± 64 | -158 ± 63 | -344 ± 63 | -189 ± 64 | -217 ± 57 | -240 ± 79 | -244 ± 82 | -249 ± 84 | -274 ± 67 | -301 ± 68 | -277 ± 68 |
| ppo_s1043 | +246 ± 56 | +70 ± 58 | +124 ± 57 | -- |  |  | -8 ± 62 | -40 ± 64 | +37 ± 61 | +76 ± 50 | +165 ± 50 | +54 ± 49 | -165 ± 50 | -83 ± 47 | -165 ± 46 | +34 ± 58 | -31 ± 59 | +35 ± 59 | -76 ± 52 | -18 ± 53 | -68 ± 53 |
| ppo_s1042 | +65 ± 50 | -3 ± 53 | -26 ± 53 |  | -- |  | -111 ± 55 | -24 ± 55 | -0 ± 54 | +1 ± 42 | +19 ± 43 | -35 ± 42 | -290 ± 46 | -203 ± 43 | -285 ± 42 | -5 ± 52 | -62 ± 53 | -47 ± 52 | -113 ± 44 | -132 ± 46 | -90 ± 45 |
| ppo_s1044 | +92 ± 60 | +52 ± 61 | +126 ± 62 |  |  | -- | +1 ± 63 | -26 ± 62 | +95 ± 60 | +90 ± 47 | +109 ± 47 | +105 ± 47 | -152 ± 50 | -153 ± 49 | -157 ± 47 | +95 ± 58 | +96 ± 58 | +48 ± 57 | -30 ± 49 | -16 ± 49 | -26 ± 50 |
| nfsp_s1042 | +100 ± 80 | +147 ± 85 | +116 ± 85 | +8 ± 62 | +111 ± 55 | -1 ± 63 | -- |  |  | -151 ± 68 | -64 ± 67 | -194 ± 68 | -124 ± 66 | -153 ± 67 | -251 ± 63 | -197 ± 82 | -337 ± 87 | -352 ± 93 | -183 ± 70 | -178 ± 68 | -146 ± 69 |
| nfsp_s1043 | +123 ± 82 | +241 ± 91 | +283 ± 89 | +40 ± 64 | +24 ± 55 | +26 ± 62 |  | -- |  | -136 ± 71 | -124 ± 72 | -160 ± 71 | -44 ± 65 | -159 ± 69 | -317 ± 64 | -147 ± 85 | -120 ± 90 | -210 ± 96 | -94 ± 72 | -142 ± 71 | -63 ± 70 |
| nfsp_s1044 | +63 ± 81 | +183 ± 88 | +84 ± 87 | -37 ± 61 | +0 ± 54 | -95 ± 60 |  |  | -- | -254 ± 68 | -126 ± 69 | -294 ± 69 | -79 ± 66 | -122 ± 66 | -323 ± 60 | -220 ± 84 | -305 ± 91 | -317 ± 94 | -267 ± 68 | -217 ± 68 | -164 ± 67 |
| deepcfr_s1042 | +178 ± 61 | +53 ± 62 | +142 ± 63 | -76 ± 50 | -1 ± 42 | -90 ± 47 | +151 ± 68 | +136 ± 71 | +254 ± 68 | -- |  |  | -196 ± 51 | -160 ± 50 | -166 ± 46 | +138 ± 62 | +134 ± 63 | +90 ± 67 | -135 ± 43 | -127 ± 48 | -104 ± 46 |
| deepcfr_s1044 | +155 ± 62 | +116 ± 64 | +116 ± 64 | -165 ± 50 | -19 ± 43 | -109 ± 47 | +64 ± 67 | +124 ± 72 | +126 ± 69 |  | -- |  | -205 ± 53 | -217 ± 51 | -207 ± 48 | +112 ± 61 | +75 ± 66 | -15 ± 66 | -157 ± 46 | -176 ± 47 | -153 ± 45 |
| deepcfr_s1043 | +210 ± 60 | +143 ± 64 | +158 ± 63 | -54 ± 49 | +35 ± 42 | -105 ± 47 | +194 ± 68 | +160 ± 71 | +294 ± 69 |  |  | -- | -208 ± 50 | -181 ± 50 | -164 ± 48 | +200 ± 65 | +91 ± 65 | +77 ± 65 | -83 ± 46 | -130 ± 43 | -92 ± 45 |
| ppo_ext_s1042 | +383 ± 61 | +303 ± 63 | +344 ± 63 | +165 ± 50 | +290 ± 46 | +152 ± 50 | +124 ± 66 | +44 ± 65 | +79 ± 66 | +196 ± 51 | +205 ± 53 | +208 ± 50 | -- |  |  | +217 ± 62 | +190 ± 62 | +136 ± 63 | +108 ± 51 | +128 ± 53 | +81 ± 53 |
| ppo_ext_s1043 | +296 ± 57 | +154 ± 59 | +189 ± 64 | +83 ± 47 | +203 ± 43 | +153 ± 49 | +153 ± 67 | +159 ± 69 | +122 ± 66 | +160 ± 50 | +217 ± 51 | +181 ± 50 |  | -- |  | +257 ± 64 | +107 ± 65 | +165 ± 65 | +62 ± 52 | +121 ± 51 | +100 ± 52 |
| ppo_ext_s1044 | +270 ± 54 | +316 ± 57 | +217 ± 57 | +165 ± 46 | +285 ± 42 | +157 ± 47 | +251 ± 63 | +317 ± 64 | +323 ± 60 | +166 ± 46 | +207 ± 48 | +164 ± 48 |  |  | -- | +310 ± 59 | +219 ± 60 | +232 ± 59 | +172 ± 49 | +135 ± 49 | +139 ± 51 |
| nfsp_ext_s1042 | +53 ± 74 | +268 ± 81 | +240 ± 79 | -34 ± 58 | +5 ± 52 | -95 ± 58 | +197 ± 82 | +147 ± 85 | +220 ± 84 | -138 ± 62 | -112 ± 61 | -200 ± 65 | -217 ± 62 | -257 ± 64 | -310 ± 59 | -- |  |  | -174 ± 64 | -194 ± 65 | -170 ± 63 |
| nfsp_ext_s1043 | +131 ± 76 | +234 ± 82 | +244 ± 82 | +31 ± 59 | +62 ± 53 | -96 ± 58 | +337 ± 87 | +120 ± 90 | +305 ± 91 | -134 ± 63 | -75 ± 66 | -91 ± 65 | -190 ± 62 | -107 ± 65 | -219 ± 60 |  | -- |  | -150 ± 65 | -134 ± 66 | -73 ± 65 |
| nfsp_ext_s1044 | +102 ± 81 | +303 ± 88 | +249 ± 84 | -35 ± 59 | +47 ± 52 | -48 ± 57 | +352 ± 93 | +210 ± 96 | +317 ± 94 | -90 ± 67 | +15 ± 66 | -77 ± 65 | -136 ± 63 | -165 ± 65 | -232 ± 59 |  |  | -- | -159 ± 67 | -57 ± 67 | -149 ± 65 |
| deepcfr_ext_s1042 | +312 ± 64 | +273 ± 66 | +274 ± 67 | +76 ± 52 | +113 ± 44 | +30 ± 49 | +183 ± 70 | +94 ± 72 | +267 ± 68 | +135 ± 43 | +157 ± 46 | +83 ± 46 | -108 ± 51 | -62 ± 52 | -172 ± 49 | +174 ± 64 | +150 ± 65 | +159 ± 67 | -- |  |  |
| deepcfr_ext_s1043 | +206 ± 64 | +196 ± 67 | +301 ± 68 | +18 ± 53 | +132 ± 46 | +16 ± 49 | +178 ± 68 | +142 ± 71 | +217 ± 68 | +127 ± 48 | +176 ± 47 | +130 ± 43 | -128 ± 53 | -121 ± 51 | -135 ± 49 | +194 ± 65 | +134 ± 66 | +57 ± 67 |  | -- |  |
| deepcfr_ext_s1044 | +286 ± 64 | +249 ± 66 | +277 ± 68 | +68 ± 53 | +90 ± 45 | +26 ± 50 | +146 ± 69 | +63 ± 70 | +164 ± 67 | +104 ± 46 | +153 ± 45 | +92 ± 45 | -81 ± 53 | -100 ± 52 | -139 ± 51 | +170 ± 63 | +73 ± 65 | +149 ± 65 |  |  | -- |

### Head-to-head by algorithm (mean over all seed pairs)

| algorithm | dqn | ppo | ppo_ext | nfsp | nfsp_ext | deepcfr | deepcfr_ext |
|---|---|---|---|---|---|---|---|
| dqn | -- | -83 ± 52 (sd 80) | -275 ± 48 (sd 75) | -149 ± 52 (sd 74) | -203 ± 61 (sd 85) | -141 ± 34 (sd 44) | -264 ± 30 (sd 40) |
| ppo | +83 ± 52 (sd 80) | -- | -184 ± 45 (sd 67) | -8 ± 39 (sd 56) | +18 ± 41 (sd 58) | +65 ± 42 (sd 62) | -63 ± 31 (sd 43) |
| ppo_ext | +275 ± 48 (sd 75) | +184 ± 45 (sd 67) | -- | +175 ± 67 (sd 100) | +204 ± 44 (sd 62) | +189 ± 21 (sd 22) | +116 ± 24 (sd 33) |
| nfsp | +149 ± 52 (sd 74) | +8 ± 39 (sd 56) | -175 ± 67 (sd 100) | -- | -245 ± 61 (sd 85) | -167 ± 49 (sd 71) | -162 ± 45 (sd 61) |
| nfsp_ext | +203 ± 61 (sd 85) | -18 ± 41 (sd 58) | -204 ± 44 (sd 62) | +245 ± 61 (sd 85) | -- | -100 ± 42 (sd 59) | -140 ± 31 (sd 46) |
| deepcfr | +141 ± 34 (sd 44) | -65 ± 42 (sd 62) | -189 ± 21 (sd 22) | +167 ± 49 (sd 71) | +100 ± 42 (sd 59) | -- | -128 ± 25 (sd 31) |
| deepcfr_ext | +264 ± 30 (sd 40) | +63 ± 31 (sd 43) | -116 ± 24 (sd 33) | +162 ± 45 (sd 61) | +140 ± 31 (sd 46) | +128 ± 25 (sd 31) | -- |

Each cell averages every agent of the row algorithm against every agent of the column algorithm; ± is a standard error that counts the training seeds of both algorithms as the random units (seed pairs share agents, so they are not independent) plus the card noise; sd is the spread over seed pairs.

### Head-to-head by position: row algorithm as small blind / as big blind

| algorithm | dqn | ppo | ppo_ext | nfsp | nfsp_ext | deepcfr | deepcfr_ext |
|---|---|---|---|---|---|---|---|
| dqn | -- | -123 / -43 | -335 / -215 | -259 / -39 | -303 / -102 | -173 / -110 | -313 / -215 |
| ppo | +43 / +123 | -- | -171 / -197 | -4 / -13 | -46 / +82 | +83 / +46 | -79 / -48 |
| ppo_ext | +215 / +335 | +197 / +171 | -- | +260 / +90 | +186 / +222 | +212 / +167 | +101 / +132 |
| nfsp | +39 / +259 | +13 / +4 | -90 / -260 | -- | -267 / -223 | -135 / -199 | -183 / -140 |
| nfsp_ext | +102 / +303 | -82 / +46 | -222 / -186 | +223 / +267 | -- | -149 / -51 | -183 / -97 |
| deepcfr | +110 / +173 | -46 / -83 | -167 / -212 | +199 / +135 | +51 / +149 | -- | -150 / -107 |
| deepcfr_ext | +215 / +313 | +48 / +79 | -132 / -101 | +140 / +183 | +97 / +183 | +107 / +150 | -- |

Mean over seed pairs, mbb/hand; the small blind acts first pre-flop and last after the flop. Each deal is played once from each position, so the overall win rate is the average of the two numbers.

### How the hands end (first algorithm's view)

| match | showdown | first folds | second folds | first wins at showdown | first wins without showdown |
|---|---|---|---|---|---|
| dqn vs ppo | 29% | 35% | 36% | +88 | -171 |
| dqn vs ppo_ext | 31% | 36% | 33% | -76 | -199 |
| dqn vs nfsp | 38% | 37% | 25% | +862 | -1011 |
| dqn vs nfsp_ext | 39% | 39% | 22% | +691 | -894 |
| dqn vs deepcfr | 35% | 32% | 33% | +319 | -461 |
| dqn vs deepcfr_ext | 31% | 36% | 33% | +332 | -596 |
| ppo vs ppo_ext | 31% | 37% | 32% | -64 | -119 |
| ppo vs nfsp | 38% | 38% | 24% | +635 | -643 |
| ppo vs nfsp_ext | 36% | 38% | 26% | +470 | -452 |
| ppo vs deepcfr | 33% | 30% | 36% | +218 | -153 |
| ppo vs deepcfr_ext | 30% | 36% | 33% | +269 | -333 |
| ppo_ext vs nfsp | 38% | 37% | 25% | +885 | -710 |
| ppo_ext vs nfsp_ext | 38% | 37% | 26% | +698 | -494 |
| ppo_ext vs deepcfr | 34% | 29% | 37% | +314 | -124 |
| ppo_ext vs deepcfr_ext | 32% | 33% | 34% | +345 | -229 |
| nfsp vs nfsp_ext | 52% | 24% | 24% | -377 | +132 |
| nfsp vs deepcfr | 43% | 21% | 36% | -441 | +274 |
| nfsp vs deepcfr_ext | 42% | 23% | 35% | -516 | +354 |
| nfsp_ext vs deepcfr | 43% | 21% | 36% | -249 | +149 |
| nfsp_ext vs deepcfr_ext | 41% | 24% | 35% | -300 | +160 |
| deepcfr vs deepcfr_ext | 37% | 35% | 28% | -23 | -105 |

Shares of hands, and the first algorithm's win rate (mbb/hand) split into the part won at showdown and the part won when a player folded (the two parts add up to the head-to-head win rate). Mean over seed pairs.

### Playing style in each matchup (row algorithm)

| algorithm | opponent | fold facing bet | raise when legal | VPIP | PFR | AF post-flop |
|---|---|---|---|---|---|---|
| dqn | ppo | 16% | 41% | 77% | 52% | 2.20 |
| dqn | ppo_ext | 15% | 40% | 78% | 51% | 2.13 |
| dqn | nfsp | 13% | 42% | 74% | 51% | 1.64 |
| dqn | nfsp_ext | 14% | 41% | 73% | 52% | 1.76 |
| dqn | deepcfr | 15% | 39% | 69% | 47% | 2.15 |
| dqn | deepcfr_ext | 15% | 41% | 74% | 49% | 1.71 |
| ppo | dqn | 17% | 45% | 73% | 43% | 2.04 |
| ppo | ppo_ext | 18% | 42% | 70% | 41% | 1.43 |
| ppo | nfsp | 16% | 39% | 68% | 42% | 0.96 |
| ppo | nfsp_ext | 17% | 41% | 68% | 46% | 1.16 |
| ppo | deepcfr | 17% | 44% | 64% | 41% | 1.75 |
| ppo | deepcfr_ext | 18% | 38% | 66% | 40% | 1.00 |
| ppo_ext | dqn | 16% | 45% | 73% | 45% | 2.34 |
| ppo_ext | ppo | 18% | 45% | 69% | 44% | 2.01 |
| ppo_ext | nfsp | 16% | 41% | 67% | 44% | 1.20 |
| ppo_ext | nfsp_ext | 17% | 43% | 68% | 49% | 1.37 |
| ppo_ext | deepcfr | 18% | 46% | 63% | 44% | 1.94 |
| ppo_ext | deepcfr_ext | 17% | 42% | 66% | 42% | 1.29 |
| nfsp | dqn | 10% | 56% | 70% | 48% | 3.16 |
| nfsp | ppo | 12% | 53% | 66% | 45% | 3.12 |
| nfsp | ppo_ext | 12% | 53% | 66% | 45% | 2.88 |
| nfsp | nfsp_ext | 8% | 55% | 64% | 44% | 2.21 |
| nfsp | deepcfr | 11% | 52% | 60% | 40% | 2.85 |
| nfsp | deepcfr_ext | 10% | 53% | 63% | 42% | 2.47 |
| nfsp_ext | dqn | 8% | 49% | 78% | 34% | 3.21 |
| nfsp_ext | ppo | 12% | 45% | 73% | 33% | 3.02 |
| nfsp_ext | ppo_ext | 12% | 46% | 74% | 32% | 2.81 |
| nfsp_ext | nfsp | 8% | 50% | 71% | 31% | 2.12 |
| nfsp_ext | deepcfr | 10% | 46% | 66% | 29% | 2.91 |
| nfsp_ext | deepcfr_ext | 11% | 46% | 70% | 30% | 2.36 |
| deepcfr | dqn | 18% | 38% | 65% | 24% | 1.91 |
| deepcfr | ppo | 21% | 34% | 61% | 23% | 1.35 |
| deepcfr | ppo_ext | 20% | 34% | 62% | 22% | 1.29 |
| deepcfr | nfsp | 17% | 35% | 59% | 23% | 1.08 |
| deepcfr | nfsp_ext | 18% | 34% | 58% | 27% | 1.09 |
| deepcfr | deepcfr_ext | 19% | 31% | 58% | 22% | 0.96 |
| deepcfr_ext | dqn | 18% | 50% | 66% | 40% | 2.46 |
| deepcfr_ext | ppo | 21% | 50% | 62% | 38% | 2.70 |
| deepcfr_ext | ppo_ext | 20% | 49% | 62% | 38% | 2.11 |
| deepcfr_ext | nfsp | 16% | 45% | 60% | 38% | 1.29 |
| deepcfr_ext | nfsp_ext | 17% | 45% | 60% | 40% | 1.40 |
| deepcfr_ext | deepcfr | 19% | 49% | 56% | 37% | 2.36 |

Sampled-action rates over all head-to-head hands of the pair: fold when facing a bet, raise when raising is legal, VPIP / PFR (share of hands with a voluntary call or raise / a raise pre-flop) and the post-flop aggression factor (bets and raises per call). Mean over seed pairs.

### Playing style in each matchup, equity-based statistics (one seed per algorithm)

| algorithm | opponent | VPIP | PFR | 3-bet | c-bet | AF | WTSD | W$SD | bluff rate | fold facing bet | fold when behind | fold when well ahead |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepcfr | deepcfr_ext | 60% | 20% | 11% | 89% | 1.08 | 62% | 50% | 21% | 20% | 23% | 15% |
| deepcfr | dqn | 64% | 22% | 11% | 88% | 2.30 | 43% | 39% | 35% | 20% | 27% | 15% |
| deepcfr | nfsp | 59% | 22% | 11% | 90% | 1.24 | 69% | 53% | 27% | 18% | 22% | 13% |
| deepcfr | nfsp_ext | 58% | 25% | 11% | 87% | 1.27 | 57% | 52% | 25% | 18% | 33% | 13% |
| deepcfr | ppo | 64% | 22% | 11% | 89% | 1.44 | 51% | 45% | 27% | 22% | 33% | 17% |
| deepcfr | ppo_ext | 64% | 19% | 11% | 89% | 1.16 | 58% | 47% | 26% | 20% | 30% | 15% |
| deepcfr_ext | deepcfr | 57% | 37% | 26% | 89% | 2.11 | 62% | 50% | 36% | 20% | 23% | 14% |
| deepcfr_ext | dqn | 64% | 38% | 29% | 92% | 3.01 | 40% | 41% | 42% | 20% | 32% | 14% |
| deepcfr_ext | nfsp | 60% | 38% | 29% | 92% | 1.49 | 72% | 52% | 34% | 17% | 21% | 13% |
| deepcfr_ext | nfsp_ext | 60% | 40% | 28% | 90% | 1.54 | 58% | 52% | 33% | 17% | 30% | 12% |
| deepcfr_ext | ppo | 64% | 38% | 29% | 91% | 2.50 | 46% | 42% | 36% | 21% | 31% | 15% |
| deepcfr_ext | ppo_ext | 64% | 36% | 29% | 91% | 1.85 | 55% | 44% | 35% | 20% | 23% | 15% |
| dqn | deepcfr | 70% | 45% | 23% | 28% | 1.77 | 42% | 61% | 15% | 20% | 77% | 12% |
| dqn | deepcfr_ext | 74% | 45% | 36% | 26% | 1.34 | 39% | 59% | 13% | 18% | 61% | 11% |
| dqn | nfsp | 75% | 47% | 35% | 26% | 1.43 | 48% | 60% | 20% | 15% | 66% | 10% |
| dqn | nfsp_ext | 74% | 51% | 36% | 27% | 1.50 | 46% | 59% | 17% | 16% | 71% | 10% |
| dqn | ppo | 81% | 50% | 36% | 26% | 1.66 | 38% | 51% | 16% | 18% | 77% | 10% |
| dqn | ppo_ext | 80% | 47% | 38% | 26% | 1.49 | 32% | 54% | 14% | 19% | 67% | 13% |
| nfsp | deepcfr | 61% | 40% | 49% | 70% | 2.20 | 69% | 47% | 41% | 10% | 4% | 6% |
| nfsp | deepcfr_ext | 64% | 41% | 51% | 65% | 1.88 | 72% | 48% | 38% | 10% | 5% | 5% |
| nfsp | dqn | 70% | 47% | 53% | 68% | 3.52 | 48% | 40% | 48% | 10% | 3% | 6% |
| nfsp | nfsp_ext | 64% | 44% | 51% | 71% | 2.05 | 71% | 49% | 46% | 8% | 5% | 4% |
| nfsp | ppo | 69% | 46% | 50% | 65% | 2.87 | 58% | 41% | 42% | 12% | 5% | 7% |
| nfsp | ppo_ext | 68% | 46% | 51% | 65% | 2.12 | 64% | 41% | 41% | 11% | 9% | 7% |
| nfsp_ext | deepcfr | 66% | 28% | 28% | 58% | 2.47 | 56% | 48% | 34% | 11% | 29% | 7% |
| nfsp_ext | deepcfr_ext | 71% | 28% | 32% | 60% | 2.05 | 58% | 48% | 34% | 11% | 26% | 6% |
| nfsp_ext | dqn | 77% | 33% | 34% | 58% | 3.52 | 47% | 41% | 42% | 9% | 27% | 5% |
| nfsp_ext | nfsp | 71% | 30% | 33% | 56% | 2.06 | 71% | 51% | 41% | 8% | 34% | 5% |
| nfsp_ext | ppo | 76% | 32% | 32% | 59% | 2.59 | 49% | 42% | 36% | 12% | 39% | 7% |
| nfsp_ext | ppo_ext | 76% | 31% | 33% | 56% | 2.06 | 50% | 44% | 35% | 12% | 30% | 7% |
| ppo | deepcfr | 69% | 46% | 39% | 70% | 1.67 | 51% | 55% | 35% | 17% | 59% | 11% |
| ppo | deepcfr_ext | 73% | 45% | 39% | 71% | 1.14 | 47% | 58% | 30% | 17% | 53% | 12% |
| ppo | dqn | 79% | 51% | 40% | 72% | 2.59 | 38% | 49% | 39% | 13% | 30% | 9% |
| ppo | nfsp | 73% | 48% | 41% | 72% | 1.06 | 58% | 59% | 29% | 15% | 35% | 10% |
| ppo | nfsp_ext | 73% | 51% | 40% | 70% | 1.30 | 49% | 58% | 33% | 15% | 49% | 10% |
| ppo | ppo_ext | 77% | 47% | 40% | 70% | 1.22 | 44% | 54% | 28% | 16% | 30% | 11% |
| ppo_ext | deepcfr | 68% | 51% | 19% | 75% | 1.89 | 59% | 53% | 45% | 16% | 55% | 8% |
| ppo_ext | deepcfr_ext | 70% | 46% | 21% | 76% | 1.29 | 55% | 56% | 38% | 14% | 56% | 8% |
| ppo_ext | dqn | 77% | 49% | 22% | 76% | 2.98 | 33% | 46% | 49% | 15% | 56% | 8% |
| ppo_ext | nfsp | 72% | 50% | 22% | 77% | 1.40 | 64% | 59% | 39% | 14% | 61% | 8% |
| ppo_ext | nfsp_ext | 73% | 57% | 22% | 75% | 1.57 | 50% | 56% | 42% | 16% | 61% | 9% |
| ppo_ext | ppo | 76% | 49% | 21% | 76% | 2.15 | 44% | 46% | 38% | 15% | 54% | 7% |

From behaviour_tracker.py on a subset of deals: 3-bet, c-bet, WTSD (went to showdown after seeing the flop), W$SD (won at showdown), bluff rate (bet or raise with equity < 0.35), fold when behind (equity below the pot odds) and fold when well ahead (equity 0.15 above them). Rates use the policy's action probabilities where available.

### Panel performance and exploitability lower bounds, every agent

| agent | panel min | vs random | vs station | vs maniac | vs nit | vs tag | vs lag | LBR | best lower bound | by | conservative |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dqn_s1042 | -997 | +1312 ± 27 | +228 ± 17 | -997 ± 77 | -403 ± 25 | -752 ± 32 | -903 ± 39 | +1361 ± 48 | +1361 ± 48 | lbr | 1267 |
| dqn_s1043 | -1183 | +1454 ± 30 | +240 ± 20 | -1183 ± 87 | -432 ± 28 | -795 ± 35 | -864 ± 42 | +1348 ± 52 | +1348 ± 52 | lbr | 1246 |
| dqn_s1044 | -1227 | +1371 ± 29 | +191 ± 18 | -1227 ± 83 | -429 ± 27 | -790 ± 34 | -943 ± 41 | +1367 ± 51 | +1367 ± 51 | lbr | 1267 |
| ppo_s1043 | -365 | +1224 ± 28 | +46 ± 19 | -359 ± 51 | -228 ± 25 | -332 ± 28 | -365 ± 34 | +822 ± 37 | +822 ± 37 | lbr | 749 |
| ppo_s1042 | -366 | +866 ± 26 | +91 ± 22 | -314 ± 43 | -250 ± 21 | -357 ± 27 | -366 ± 32 | +738 ± 33 | +738 ± 33 | lbr | 673 |
| ppo_s1044 | -300 | +1036 ± 29 | +168 ± 25 | -269 ± 51 | -147 ± 24 | -266 ± 31 | -300 ± 36 | +616 ± 41 | +616 ± 41 | lbr | 536 |
| nfsp_s1042 | -1075 | +1546 ± 32 | +127 ± 25 | +717 ± 86 | -647 ± 31 | -984 ± 42 | -1075 ± 48 | +1467 ± 56 | +1467 ± 56 | lbr | 1357 |
| nfsp_s1043 | -1122 | +1610 ± 33 | +114 ± 25 | +885 ± 93 | -615 ± 33 | -1029 ± 43 | -1122 ± 51 | +1653 ± 60 | +1653 ± 60 | lbr | 1535 |
| nfsp_s1044 | -1165 | +1435 ± 31 | +89 ± 25 | +716 ± 95 | -716 ± 32 | -1074 ± 46 | -1165 ± 53 | +1582 ± 62 | +1582 ± 62 | lbr | 1459 |
| deepcfr_s1042 | -397 | +977 ± 27 | +277 ± 24 | +561 ± 66 | -321 ± 24 | -393 ± 32 | -397 ± 37 | +630 ± 37 | +630 ± 37 | lbr | 557 |
| deepcfr_s1044 | -434 | +981 ± 27 | +266 ± 24 | +461 ± 66 | -356 ± 24 | -434 ± 32 | -431 ± 37 | +578 ± 38 | +578 ± 38 | lbr | 504 |
| deepcfr_s1043 | -437 | +1067 ± 28 | +270 ± 24 | +506 ± 68 | -344 ± 25 | -428 ± 33 | -437 ± 37 | +557 ± 39 | +557 ± 39 | lbr | 481 |
| ppo_ext_s1042 | -230 | +1335 ± 30 | +186 ± 22 | +76 ± 54 | -195 ± 27 | -230 ± 30 | -140 ± 33 | +567 ± 41 | +567 ± 41 | lbr | 486 |
| ppo_ext_s1043 | -218 | +1091 ± 28 | +251 ± 21 | -190 ± 59 | -125 ± 25 | -218 ± 31 | -177 ± 35 | +676 ± 46 | +676 ± 46 | lbr | 585 |
| ppo_ext_s1044 | -306 | +1089 ± 27 | +178 ± 22 | +112 ± 53 | -218 ± 22 | -306 ± 28 | -198 ± 33 | +382 ± 38 | +382 ± 38 | lbr | 308 |
| nfsp_ext_s1042 | -990 | +1507 ± 32 | +245 ± 23 | +617 ± 85 | -614 ± 30 | -936 ± 39 | -990 ± 45 | +1194 ± 51 | +1194 ± 51 | lbr | 1095 |
| nfsp_ext_s1043 | -960 | +1541 ± 32 | +264 ± 23 | +820 ± 86 | -577 ± 31 | -928 ± 40 | -960 ± 46 | +1298 ± 52 | +1298 ± 52 | lbr | 1196 |
| nfsp_ext_s1044 | -1157 | +1546 ± 32 | +303 ± 23 | +816 ± 94 | -584 ± 32 | -1023 ± 42 | -1157 ± 48 | +1310 ± 55 | +1310 ± 55 | lbr | 1202 |
| deepcfr_ext_s1042 | -310 | +1193 ± 29 | +320 ± 26 | +358 ± 62 | -254 ± 27 | -310 ± 34 | -278 ± 38 | +509 ± 41 | +509 ± 41 | lbr | 428 |
| deepcfr_ext_s1043 | -287 | +1216 ± 30 | +352 ± 25 | +268 ± 62 | -262 ± 27 | -287 ± 33 | -275 ± 37 | +515 ± 41 | +515 ± 41 | lbr | 434 |
| deepcfr_ext_s1044 | -320 | +1145 ± 29 | +323 ± 26 | +335 ± 62 | -267 ± 26 | -320 ± 33 | -316 ± 37 | +545 ± 41 | +545 ± 41 | lbr | 465 |

Panel columns: the agent's win rate against each rule-based bot (panel min = the worst of them, excluding random). LBR / best lower bound: what a counter-strategy wins against the agent (higher = more exploitable); conservative = largest 95% lower confidence bound, floored at 0.

### Summary by algorithm (mean over seeds, sd = spread over seeds)

| algorithm | agents | panel min | vs random | vs station | vs maniac | vs nit | vs tag | vs lag | LBR | best lower bound |
|---|---|---|---|---|---|---|---|---|---|---|
| dqn | 3 | -1136 (sd 122) | +1379 (sd 71) | +219 (sd 26) | -1136 (sd 122) | -421 (sd 16) | -779 (sd 23) | -903 (sd 39) | +1359 (sd 9) | +1359 (sd 9) |
| ppo | 3 | -344 (sd 38) | +1042 (sd 179) | +102 (sd 62) | -314 (sd 45) | -208 (sd 54) | -318 (sd 47) | -344 (sd 38) | +725 (sd 103) | +725 (sd 103) |
| ppo_ext | 3 | -251 (sd 48) | +1172 (sd 141) | +205 (sd 40) | -1 (sd 165) | -179 (sd 49) | -251 (sd 48) | -172 (sd 29) | +542 (sd 148) | +542 (sd 148) |
| nfsp | 3 | -1120 (sd 45) | +1530 (sd 89) | +110 (sd 19) | +773 (sd 97) | -659 (sd 52) | -1029 (sd 45) | -1120 (sd 45) | +1567 (sd 94) | +1567 (sd 94) |
| nfsp_ext | 3 | -1036 (sd 106) | +1531 (sd 21) | +271 (sd 30) | +751 (sd 116) | -591 (sd 20) | -963 (sd 53) | -1036 (sd 106) | +1267 (sd 64) | +1267 (sd 64) |
| deepcfr | 3 | -423 (sd 23) | +1008 (sd 51) | +271 (sd 6) | +509 (sd 50) | -340 (sd 18) | -418 (sd 22) | -422 (sd 22) | +588 (sd 37) | +588 (sd 37) |
| deepcfr_ext | 3 | -306 (sd 17) | +1185 (sd 37) | +332 (sd 18) | +321 (sd 47) | -261 (sd 7) | -306 (sd 17) | -290 (sd 23) | +523 (sd 19) | +523 (sd 19) |

### Ranking by the head-to-head round robin

| algorithm | mean vs others | wins | significant wins |
|---|---|---|---|
| ppo_ext | +190 | 6 / 6 | 6 / 6 |
| deepcfr_ext | +107 | 5 / 6 | 5 / 6 |
| deepcfr | +4 | 3 / 6 | 3 / 6 |
| nfsp_ext | -2 | 2 / 6 | 2 / 6 |
| ppo | -15 | 3 / 6 | 0 / 6 |
| nfsp | -99 | 2 / 6 | 1 / 6 |
| dqn | -186 | 0 / 6 | 0 / 6 |

Mean = average head-to-head win rate against the other algorithms (mbb/hand); wins = opponents beaten on average; significant = beaten by more than 2 standard errors. Intransitive cycles among algorithms: ppo > nfsp_ext > nfsp > ppo; ppo > deepcfr > nfsp > ppo Agent level: 34 of 945 triples of agents from three different algorithms (4%) form a cycle.

### Rank correlation (Spearman) between the measures, over 21 agents

| measure | head-to-head | panel min | -LBR | -best lower bound |
|---|---|---|---|---|
| head-to-head | -- | +0.82 | +0.82 | +0.82 |
| panel min | +0.82 | -- | +0.80 | +0.80 |
| -LBR | +0.82 | +0.80 | -- | +1.00 |
| -best lower bound | +0.82 | +0.80 | +1.00 | -- |

Each measure oriented so that higher = better (exploitability bounds negated); head-to-head = the agent's mean win rate against the agents of the other algorithms. +1 = the measures rank the agents identically.
