# speed-4b

130 rounds per fused token; each device contributes 1,382,912 values per token (2.77 MB in bf16), of which 248,320 are output logits. One child is 1.84 GB.

| Child | Agreement with the group | Tokens per cycle (k = 1, 4, 8) |
|---:|---:|---|
| 0 | 0.790 | 1.79, 3.27, 4.14 |
| 1 | 0.790 | 1.79, 3.27, 4.15 |
| 2 | 0.795 | 1.79, 3.31, 4.21 |
| 3 | 0.792 | 1.79, 3.28, 4.15 |

| Latency | Link | Memory | Solo tok/s | Fused tok/s | Draft | Fused / solo |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 ms | 1 Gbit/s | 50 GB/s | 27.2 | 12.5 | 0 | 46% |
| 0.05 ms | 1 Gbit/s | 100 GB/s | 54.3 | 16.2 | 0 | 30% |
| 0.05 ms | 1 Gbit/s | 1000 GB/s | 543.0 | 23.2 | 1 | 4% |
| 0.05 ms | 10 Gbit/s | 50 GB/s | 27.2 | 20.4 | 0 | 75% |
| 0.05 ms | 10 Gbit/s | 100 GB/s | 54.3 | 33.4 | 1 | 62% |
| 0.05 ms | 10 Gbit/s | 1000 GB/s | 543.0 | 94.0 | 2 | 17% |
| 0.2 ms | 1 Gbit/s | 50 GB/s | 27.2 | 9.9 | 1 | 36% |
| 0.2 ms | 1 Gbit/s | 100 GB/s | 54.3 | 12.4 | 1 | 23% |
| 0.2 ms | 1 Gbit/s | 1000 GB/s | 543.0 | 16.3 | 2 | 3% |
| 0.2 ms | 10 Gbit/s | 50 GB/s | 27.2 | 16.0 | 1 | 59% |
| 0.2 ms | 10 Gbit/s | 100 GB/s | 54.3 | 24.3 | 2 | 45% |
| 0.2 ms | 10 Gbit/s | 1000 GB/s | 543.0 | 51.5 | 4 | 9% |
| 1 ms | 1 Gbit/s | 50 GB/s | 27.2 | 6.0 | 3 | 22% |
| 1 ms | 1 Gbit/s | 100 GB/s | 54.3 | 7.1 | 4 | 13% |
| 1 ms | 1 Gbit/s | 1000 GB/s | 543.0 | 8.7 | 4 | 2% |
| 1 ms | 10 Gbit/s | 50 GB/s | 27.2 | 9.6 | 3 | 35% |
| 1 ms | 10 Gbit/s | 100 GB/s | 54.3 | 13.0 | 4 | 24% |
| 1 ms | 10 Gbit/s | 1000 GB/s | 543.0 | 20.8 | 7 | 4% |

**H9: fail.** At 0.2 ms, 1 Gbit/s and 100 GB/s, four fused devices reach 12.4 tokens/s against 54.3 solo (23%; at least 25% needed), drafting 1 tokens per cycle with child 1.
