### Training cost of the winning configurations

| game | algorithm | study | budget | runs | hours per run | parameters (playing) | parameters (all trained) | tuning trials (complete / pruned / failed) | tuning worker-hours |
|---|---|---|---|---|---|---|---|---|---|
| limit | DQN | dqn_v1 | 1,000,000 episodes | 3 | 8.0 (sd 1.3) | 106,857 | 106,857 | 5 / 45 / 0 | 182 |
| limit | PPO | ppo_v2 | 1,000,000 episodes | 3 | 1.1 (sd 0.5) | 106,501 | 106,501 | 7 / 43 / 0 | 12 |
| limit | PPO extended | ppo_v2 | 4,000,000 episodes | 3 | 11.3 (sd 0.1) | 106,501 | 106,501 | - | - |
| limit | NFSP | nfsp_v3 | 2,000,000 episodes | 3 | 1.8 (sd 0.2) | 106,857 | 213,714 | 9 / 41 / 0 | 49 |
| limit | NFSP extended | nfsp_v3 | 8,000,000 episodes | 3 | 8.6 (sd 2.5) | 106,857 | 213,714 | - | - |
| limit | Deep CFR | deepcfr_v1 | 200 iterations | 3 | 6.6 (sd 0.2) | 53,252 | 159,756 | 4 / 26 / 0 | 104 |
| limit | Deep CFR extended | deepcfr_v1 | 500 iterations | 3 | 19.8 (sd 0.2) | 53,252 | 159,756 | - | - |
| leduc | DQN | leduc_dqn_v1 | 1,000,000 episodes | 3 | 1.7 (sd 0.0) | 76,437 | 76,437 | 9 / 21 / 0 | 41 |
| leduc | PPO | leduc_ppo_v1 | 1,000,000 episodes | 3 | 0.3 (sd 0.0) | 76,549 | 76,549 | 5 / 25 / 0 | 4 |
| leduc | NFSP | leduc_nfsp_v1 | 2,000,000 episodes | 3 | 0.8 (sd 0.0) | 21,909 | 43,818 | 7 / 23 / 0 | 13 |
| leduc | Deep CFR | leduc_deepcfr_v1 | 300 iterations | 3 | 0.3 (sd 0.0) | 21,764 | 65,292 | 7 / 23 / 0 | 9 |

Hours per run: wall time of one training run (3-4 CPU threads, 4 runs in parallel on a laptop), mean and sd over seeds. Parameters: the network that plays (DQN: Q-network; PPO: actor-critic network; NFSP: average-policy network; Deep CFR: policy network) and every network trained (NFSP adds its best-response Q-network, Deep CFR its two advantage networks); optimizer states and target networks are not counted. Tuning worker-hours: summed wall time of all trials of the study.
