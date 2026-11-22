# Reward Sensitivity - v2_nep

Bu tablo ayni test trajectories uzerindeki reward components'i yeniden agirliklandirir; PPO yeniden egitilmemistir. Bu nedenle scalar cost ranking robustness kanitidir, training sensitivity iddiasi degildir.

| Weight profile | En dusuk cost kosulu | Cost | PPO-reference rank | Greedy-reference rank | PPO - greedy cost [%95 GA] |
|---|---|---:|---:|---:|---:|
| base | greedy_semantic[reference] | 0.887 | 3 | 1 | +0.131 [+0.075, +0.205] |
| latency_norm x0.5 | greedy_semantic[reference] | 0.704 | 2 | 1 | +0.085 [+0.044, +0.138] |
| latency_norm x2 | greedy_semantic[reference] | 1.254 | 3 | 1 | +0.223 [+0.133, +0.343] |
| energy_norm x0.5 | greedy_semantic[reference] | 0.854 | 2 | 1 | +0.125 [+0.074, +0.189] |
| energy_norm x2 | greedy_semantic[reference] | 0.954 | 3 | 1 | +0.144 [+0.077, +0.239] |
| deadline_miss x0.5 | greedy_semantic[reference] | 0.796 | 3 | 1 | +0.131 [+0.078, +0.202] |
| deadline_miss x2 | greedy_semantic[reference] | 1.070 | 2 | 1 | +0.131 [+0.064, +0.213] |
| soft_constraint_violation x0.5 | greedy_semantic[reference] | 0.760 | 3 | 1 | +0.115 [+0.059, +0.189] |
| soft_constraint_violation x2 | greedy_semantic[reference] | 1.141 | 2 | 1 | +0.164 [+0.103, +0.241] |
| failure_or_drop x0.5 | greedy_semantic[reference] | 0.878 | 3 | 1 | +0.134 [+0.079, +0.207] |
| failure_or_drop x2 | greedy_semantic[reference] | 0.905 | 2 | 1 | +0.125 [+0.064, +0.200] |
| hard_constraint_violation x0.5 | greedy_semantic[reference] | 0.887 | 3 | 1 | +0.131 [+0.075, +0.205] |
| hard_constraint_violation x2 | greedy_semantic[reference] | 0.887 | 3 | 1 | +0.131 [+0.075, +0.205] |

Base weights: `{"deadline_miss": 5.0, "energy_norm": 1.0, "failure_or_drop": 10.0, "hard_constraint_violation": 20.0, "latency_norm": 1.0, "soft_constraint_violation": 5.0}`.
