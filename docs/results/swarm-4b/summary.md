# swarm-4b

## GSM8K

| Condition | Accuracy | Correct |
|---|---:|---:|
| bf16 | 0.8256 | 1089/1319 |
| child/k0 | 0.7104 | 937/1319 |
| child/k1 | 0.6801 | 897/1319 |
| child/k2 | 0.7066 | 932/1319 |
| child/k3 | 0.7422 | 979/1319 |
| drop/k0 | 0.8287 | 1093/1319 |
| drop/k1 | 0.8309 | 1096/1319 |
| drop/k2 | 0.8279 | 1092/1319 |
| drop/k3 | 0.8256 | 1089/1319 |
| group/k0,1,2,3 | 0.8287 | 1093/1319 |
| group/k0,1,2 | 0.8044 | 1061/1319 |
| group/k0,1,3 | 0.8120 | 1071/1319 |
| group/k0,1 | 0.7657 | 1010/1319 |
| group/k0,2,3 | 0.8324 | 1098/1319 |
| group/k0,2 | 0.8074 | 1065/1319 |
| group/k0,3 | 0.7892 | 1041/1319 |
| group/k1,2,3 | 0.8089 | 1067/1319 |
| group/k1,2 | 0.7839 | 1034/1319 |
| group/k1,3 | 0.8097 | 1068/1319 |
| group/k2,3 | 0.7900 | 1042/1319 |

### Pre-registered checks

- **H7: pass.** worst child 0.680, fused 0.829, bf16 0.826; gain +14.9 points, 102% of the gap, McNemar p = 5.2e-32
- **H8 (child 0, reference): fail.** child 0: escalating 30% gives 0.766 from 0.710, 47% of the +11.8-point gain (random: 30%)
- **H8: fail.** child 1: escalating 30% gives 0.743 from 0.680, 42% of the +14.9-point gain (random: 30%)
- **H8 (child 2, reference): pass.** child 2: escalating 30% gives 0.768 from 0.707, 50% of the +12.2-point gain (random: 30%)
- **H8 (child 3, reference): pass.** child 3: escalating 30% gives 0.791 from 0.742, 56% of the +8.6-point gain (random: 30%)
- **H11: pass.** mean accuracy by group size 1: 0.710, 2: 0.791, 3: 0.814, 4: 0.829; worst group of 3 0.804 vs best child 0.742
- **H13: pass.** mean difference +1.38 points, one-sided 95% lower bound +0.58 (margin -2)

## MMLU

| Condition | Accuracy | Correct |
|---|---:|---:|
| bf16 | 0.7296 | 10245/14042 |
| child/k0 | 0.6724 | 9442/14042 |
| child/k1 | 0.6566 | 9220/14042 |
| child/k2 | 0.7008 | 9840/14042 |
| child/k3 | 0.6677 | 9376/14042 |
| group/k0,1,2,3 | 0.7284 | 10228/14042 |
| group/k0,1,2 | 0.7243 | 10171/14042 |
| group/k0,1,3 | 0.7262 | 10197/14042 |
| group/k0,1 | 0.7062 | 9916/14042 |
| group/k0,2,3 | 0.7219 | 10137/14042 |
| group/k0,2 | 0.7190 | 10096/14042 |
| group/k0,3 | 0.7107 | 9980/14042 |
| group/k1,2,3 | 0.7290 | 10237/14042 |
| group/k1,2 | 0.7147 | 10036/14042 |
| group/k1,3 | 0.7253 | 10185/14042 |
| group/k2,3 | 0.7084 | 9947/14042 |
