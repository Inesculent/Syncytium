Worst-case perplexity with *s* of *N* nodes alive. *replicated*: every node holds the same round-to-nearest model (same memory as a child); `replicated_match_bpw` is the extra memory per node at which it becomes at least as good as staggered. *coded*: every node holds one base child plus an erasure-coded share of the parent's remaining bits, any *s* of which rebuild the parent; `coded_extra_bpw` is its extra memory per node. Quantized weights are 100.0% of all parameters. Blank staggered cells were not measured.

|   bits |   group_size |   phase | rotation   | scaling   |   num_children |   survivors |   staggered |   child_bpw |   replicated |   replicated_match_bpw |   coded |   coded_extra_bpw |
|-------:|-------------:|--------:|:-----------|:----------|---------------:|------------:|------------:|------------:|-------------:|-----------------------:|--------:|------------------:|
|      3 |           64 |     0.5 | on         | off       |              4 |           1 |       9.225 |       3.501 |        9.053 |                      0 |   7.058 |             2     |
|      3 |           64 |     0.5 | on         | off       |              4 |           2 |     nan     |       3.501 |        9.053 |                    nan |   7.058 |             1     |
|      3 |           64 |     0.5 | on         | off       |              4 |           3 |       7.22  |       3.501 |        9.053 |                    nan |   7.058 |             0.667 |
|      3 |           64 |     0.5 | on         | off       |              4 |           4 |       7.058 |       3.501 |        9.053 |                    nan |   7.058 |             0.5   |
|      3 |           64 |     0.5 | on         | on        |              4 |           1 |       7.921 |       3.501 |        7.828 |                      0 |   7.031 |             2     |
|      3 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       3.501 |        7.828 |                    nan |   7.031 |             1     |
|      3 |           64 |     0.5 | on         | on        |              4 |           3 |       7.109 |       3.501 |        7.828 |                    nan |   7.031 |             0.667 |
|      3 |           64 |     0.5 | on         | on        |              4 |           4 |       7.031 |       3.501 |        7.828 |                    nan |   7.031 |             0.5   |
|      4 |           64 |     0.5 | on         | off       |              4 |           1 |       7.33  |       4.501 |        7.283 |                      0 |   7.003 |             2     |
|      4 |           64 |     0.5 | on         | off       |              4 |           2 |     nan     |       4.501 |        7.283 |                    nan |   7.003 |             1     |
|      4 |           64 |     0.5 | on         | off       |              4 |           3 |       7.037 |       4.501 |        7.283 |                    nan |   7.003 |             0.667 |
|      4 |           64 |     0.5 | on         | off       |              4 |           4 |       7.003 |       4.501 |        7.283 |                    nan |   7.003 |             0.5   |
|      4 |           64 |     0.5 | on         | on        |              4 |           1 |       7.163 |       4.501 |        7.15  |                      0 |   6.994 |             2     |
|      4 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       4.501 |        7.15  |                    nan |   6.994 |             1     |
|      4 |           64 |     0.5 | on         | on        |              4 |           3 |       7.011 |       4.501 |        7.15  |                    nan |   6.994 |             0.667 |
|      4 |           64 |     0.5 | on         | on        |              4 |           4 |       6.994 |       4.501 |        7.15  |                    nan |   6.994 |             0.5   |
