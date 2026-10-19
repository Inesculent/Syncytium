# speed-9b

130 rounds per fused token; each device contributes 1,679,360 values per token (3.36 MB in bf16), of which 248,320 are output logits. One child is 3.92 GB.

| Child | Agreement with the group | Tokens per cycle (k = 1, 4, 8) |
|---:|---:|---|
| 0 | 0.830 | 1.82, 3.55, 4.72 |
| 1 | 0.832 | 1.83, 3.56, 4.75 |
| 2 | 0.827 | 1.82, 3.51, 4.69 |
| 3 | 0.832 | 1.82, 3.57, 4.78 |

| Latency | Link | Memory | Solo tok/s | Fused tok/s | Draft | Fused / solo |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 ms | 1 Gbit/s | 50 GB/s | 12.8 | 7.8 | 0 | 61% |
| 0.05 ms | 1 Gbit/s | 100 GB/s | 25.5 | 11.2 | 0 | 44% |
| 0.05 ms | 1 Gbit/s | 1000 GB/s | 255.2 | 19.1 | 1 | 7% |
| 0.05 ms | 10 Gbit/s | 50 GB/s | 12.8 | 10.9 | 0 | 85% |
| 0.05 ms | 10 Gbit/s | 100 GB/s | 25.5 | 19.0 | 0 | 74% |
| 0.05 ms | 10 Gbit/s | 1000 GB/s | 255.2 | 72.3 | 2 | 28% |
| 0.2 ms | 1 Gbit/s | 50 GB/s | 12.8 | 6.4 | 1 | 50% |
| 0.2 ms | 1 Gbit/s | 100 GB/s | 25.5 | 8.9 | 2 | 35% |
| 0.2 ms | 1 Gbit/s | 1000 GB/s | 255.2 | 14.2 | 2 | 6% |
| 0.2 ms | 10 Gbit/s | 50 GB/s | 12.8 | 9.2 | 1 | 72% |
| 0.2 ms | 10 Gbit/s | 100 GB/s | 25.5 | 15.3 | 1 | 60% |
| 0.2 ms | 10 Gbit/s | 1000 GB/s | 255.2 | 44.6 | 4 | 17% |
| 1 ms | 1 Gbit/s | 50 GB/s | 12.8 | 4.5 | 2 | 35% |
| 1 ms | 1 Gbit/s | 100 GB/s | 25.5 | 5.8 | 3 | 23% |
| 1 ms | 1 Gbit/s | 1000 GB/s | 255.2 | 8.1 | 5 | 3% |
| 1 ms | 10 Gbit/s | 50 GB/s | 12.8 | 6.5 | 2 | 51% |
| 1 ms | 10 Gbit/s | 100 GB/s | 25.5 | 9.7 | 4 | 38% |
| 1 ms | 10 Gbit/s | 1000 GB/s | 255.2 | 20.4 | 7 | 8% |

**H9: pass.** At 0.2 ms, 1 Gbit/s and 100 GB/s, four fused devices reach 8.9 tokens/s against 25.5 solo (35%; at least 25% needed), drafting 2 tokens per cycle with child 2.
