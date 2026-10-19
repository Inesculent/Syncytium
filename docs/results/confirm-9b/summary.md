# confirm-9b

| Condition | Accuracy |
|---|---:|
| bf16 | 0.8787 |
| child/k0 | 0.7968 |
| child/k1 | 0.8317 |
| child/k2 | 0.8271 |
| child/k3 | 0.8203 |
| group/k0,1,2,3 | 0.8764 |
| rtn/b6 | 0.8795 |

## H8b

- **H8b: pass.** gain of the full group over child 0: +8.0 points; 3 of 3 partners pass
- **H8b, partner 1: pass.** child 0 asking, child 1 as partner: escalates 24%, accuracy 0.885, 110% of the gain (random escalation at the same rate: 24%)
- **H8b, partner 2: pass.** child 0 asking, child 2 as partner: escalates 23%, accuracy 0.882, 107% of the gain (random escalation at the same rate: 23%)
- **H8b, partner 3: pass.** child 0 asking, child 3 as partner: escalates 23%, accuracy 0.882, 108% of the gain (random escalation at the same rate: 23%)

## H10b

| System | Accuracy | Problems/s per answer token |
|---|---:|---:|
| swarm | 0.883 | 21.9 |
| split A: two agents + 6-bit on two devices | 0.885 | 25.5 |
| split B: 6-bit on four devices | 0.879 | 54.4 |
| floor: four lone agents | 0.819 | 102.1 |

**H10b: fail.** swarm 0.883 at 21.9; static splits at least as accurate: split A
