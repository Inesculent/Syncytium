# confirm-4b

| Condition | Accuracy |
|---|---:|
| bf16 | 0.8423 |
| child/k0 | 0.7240 |
| child/k1 | 0.6641 |
| child/k2 | 0.7384 |
| child/k3 | 0.7582 |
| group/k0,1,2,3 | 0.8294 |
| rtn/b6 | 0.8415 |

## H8b

- **H8b: pass.** gain of the full group over child 1: +16.5 points; 3 of 3 partners pass
- **H8b, partner 0: pass.** child 1 asking, child 0 as partner: escalates 35%, accuracy 0.816, 92% of the gain (random escalation at the same rate: 35%)
- **H8b, partner 2: pass.** child 1 asking, child 2 as partner: escalates 34%, accuracy 0.819, 94% of the gain (random escalation at the same rate: 34%)
- **H8b, partner 3: pass.** child 1 asking, child 3 as partner: escalates 35%, accuracy 0.830, 100% of the gain (random escalation at the same rate: 35%)

## H10b

| System | Accuracy | Problems/s per answer token |
|---|---:|---:|
| swarm | 0.822 | 26.9 |
| split A: two agents + 6-bit on two devices | 0.829 | 54.3 |
| split B: 6-bit on four devices | 0.842 | 114.6 |
| floor: four lone agents | 0.721 | 217.2 |

**H10b: fail.** swarm 0.822 at 26.9; static splits at least as accurate: split A, split B
