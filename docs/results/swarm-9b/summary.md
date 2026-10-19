# swarm-9b

## GSM8K

| Condition | Accuracy | Correct |
|---|---:|---:|
| bf16 | 0.8620 | 1137/1319 |
| child/k0 | 0.7680 | 1013/1319 |
| child/k1 | 0.7976 | 1052/1319 |
| child/k2 | 0.8180 | 1079/1319 |
| child/k3 | 0.8067 | 1064/1319 |
| drop/k0 | 0.8552 | 1128/1319 |
| drop/k1 | 0.8605 | 1135/1319 |
| drop/k2 | 0.8605 | 1135/1319 |
| drop/k3 | 0.8582 | 1132/1319 |
| group/k0,1,2,3 | 0.8605 | 1135/1319 |
| group/k0,1,2 | 0.8453 | 1115/1319 |
| group/k0,1,3 | 0.8431 | 1112/1319 |
| group/k0,1 | 0.8226 | 1085/1319 |
| group/k0,2,3 | 0.8575 | 1131/1319 |
| group/k0,2 | 0.8476 | 1118/1319 |
| group/k0,3 | 0.8218 | 1084/1319 |
| group/k1,2,3 | 0.8529 | 1125/1319 |
| group/k1,2 | 0.8544 | 1127/1319 |
| group/k1,3 | 0.8476 | 1118/1319 |
| group/k2,3 | 0.8499 | 1121/1319 |

### Pre-registered checks

- **H7: pass.** worst child 0.768, fused 0.861, bf16 0.862; gain +9.2 points, 98% of the gap, McNemar p = 4.8e-18
- **H8: fail.** child 0: escalating 30% gives 0.801 from 0.768, 36% of the +9.2-point gain (random: 30%)
- **H8 (child 1, reference): pass.** child 1: escalating 30% gives 0.832 from 0.798, 55% of the +6.3-point gain (random: 30%)
- **H8 (child 2, reference): fail.** child 2: escalating 30% gives 0.835 from 0.818, 41% of the +4.2-point gain (random: 30%)
- **H8 (child 3, reference): pass.** child 3: escalating 30% gives 0.835 from 0.807, 54% of the +5.4-point gain (random: 30%)
- **H11: pass.** mean accuracy by group size 1: 0.798, 2: 0.841, 3: 0.850, 4: 0.861; worst group of 3 0.843 vs best child 0.818
- **H13: pass.** mean difference +0.89 points, one-sided 95% lower bound +0.18 (margin -2)

## MMLU

| Condition | Accuracy | Correct |
|---|---:|---:|
| bf16 | 0.7187 | 10092/14042 |
| child/k0 | 0.6928 | 9728/14042 |
| child/k1 | 0.7004 | 9835/14042 |
| child/k2 | 0.6988 | 9813/14042 |
| child/k3 | 0.6868 | 9644/14042 |
| group/k0,1,2,3 | 0.7076 | 9936/14042 |
| group/k0,1,2 | 0.7093 | 9960/14042 |
| group/k0,1,3 | 0.7023 | 9861/14042 |
| group/k0,1 | 0.7032 | 9875/14042 |
| group/k0,2,3 | 0.7099 | 9969/14042 |
| group/k0,2 | 0.7100 | 9970/14042 |
| group/k0,3 | 0.7053 | 9904/14042 |
| group/k1,2,3 | 0.7110 | 9984/14042 |
| group/k1,2 | 0.7117 | 9994/14042 |
| group/k1,3 | 0.6977 | 9797/14042 |
| group/k2,3 | 0.7069 | 9926/14042 |
