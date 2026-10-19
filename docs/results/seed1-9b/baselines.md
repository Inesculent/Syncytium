Worst-case perplexity with *s* of *N* nodes alive. *replicated*: every node holds the same round-to-nearest model (same memory as a child); `replicated_match_bpw` is the extra memory per node at which it becomes at least as good as staggered. *coded*: every node holds one base child plus an erasure-coded share of the parent's remaining bits, any *s* of which rebuild the parent; `coded_extra_bpw` is its extra memory per node. Quantized weights are 100.0% of all parameters. Blank staggered cells were not measured.

|   bits |   group_size |   phase | rotation   | scaling   |   num_children |   survivors |   staggered |   child_bpw |   replicated |   replicated_match_bpw |   coded |   coded_extra_bpw |
|-------:|-------------:|--------:|:-----------|:----------|---------------:|------------:|------------:|------------:|-------------:|-----------------------:|--------:|------------------:|
|      3 |           64 |     0.5 | on         | off       |              4 |           1 |       9.159 |       3.501 |        8.935 |                      0 |   7.053 |             2     |
|      3 |           64 |     0.5 | on         | off       |              4 |           2 |     nan     |       3.501 |        8.935 |                    nan |   7.053 |             1     |
|      3 |           64 |     0.5 | on         | off       |              4 |           3 |       7.201 |       3.501 |        8.935 |                    nan |   7.053 |             0.667 |
|      3 |           64 |     0.5 | on         | off       |              4 |           4 |       7.053 |       3.501 |        8.935 |                    nan |   7.053 |             0.5   |
|      3 |           64 |     0.5 | on         | on        |              4 |           1 |       7.904 |       3.501 |        7.849 |                      0 |   7.025 |             2     |
|      3 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       3.501 |        7.849 |                    nan |   7.025 |             1     |
|      3 |           64 |     0.5 | on         | on        |              4 |           3 |       7.105 |       3.501 |        7.849 |                    nan |   7.025 |             0.667 |
|      3 |           64 |     0.5 | on         | on        |              4 |           4 |       7.025 |       3.501 |        7.849 |                    nan |   7.025 |             0.5   |
|      4 |           64 |     0.5 | on         | off       |              4 |           1 |       7.349 |       4.501 |        7.27  |                      0 |   7.002 |             2     |
|      4 |           64 |     0.5 | on         | off       |              4 |           2 |     nan     |       4.501 |        7.27  |                    nan |   7.002 |             1     |
|      4 |           64 |     0.5 | on         | off       |              4 |           3 |       7.039 |       4.501 |        7.27  |                    nan |   7.002 |             0.667 |
|      4 |           64 |     0.5 | on         | off       |              4 |           4 |       7.002 |       4.501 |        7.27  |                    nan |   7.002 |             0.5   |
|      4 |           64 |     0.5 | on         | on        |              4 |           1 |       7.16  |       4.501 |        7.146 |                      0 |   6.992 |             2     |
|      4 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       4.501 |        7.146 |                    nan |   6.992 |             1     |
|      4 |           64 |     0.5 | on         | on        |              4 |           3 |       7.009 |       4.501 |        7.146 |                    nan |   6.992 |             0.667 |
|      4 |           64 |     0.5 | on         | on        |              4 |           4 |       6.992 |       4.501 |        7.146 |                    nan |   6.992 |             0.5   |
