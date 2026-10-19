Worst-case perplexity with *s* of *N* nodes alive. *replicated*: every node holds the same round-to-nearest model (same memory as a child); `replicated_match_bpw` is the extra memory per node at which it becomes at least as good as staggered. *coded*: every node holds one base child plus an erasure-coded share of the parent's remaining bits, any *s* of which rebuild the parent; `coded_extra_bpw` is its extra memory per node. Quantized weights are 100.0% of all parameters. Blank staggered cells were not measured.

|   bits |   group_size |   phase | rotation   | scaling   |   num_children |   survivors |   staggered |   child_bpw |   replicated |   replicated_match_bpw |   coded |   coded_extra_bpw |
|-------:|-------------:|--------:|:-----------|:----------|---------------:|------------:|------------:|------------:|-------------:|-----------------------:|--------:|------------------:|
|      2 |           64 |     0.5 | on         | on        |              4 |           1 |      13.913 |       2.501 |       12.26  |                      0 |   7.125 |             2     |
|      2 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       2.501 |       12.26  |                    nan |   7.125 |             1     |
|      2 |           64 |     0.5 | on         | on        |              4 |           3 |       7.477 |       2.501 |       12.26  |                      1 |   7.125 |             0.667 |
|      2 |           64 |     0.5 | on         | on        |              4 |           4 |       7.125 |       2.501 |       12.26  |                      2 |   7.125 |             0.5   |
|      3 |           64 |     0.5 | on         | on        |              4 |           1 |       7.673 |       3.501 |        7.191 |                      0 |   6.944 |             2     |
|      3 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       3.501 |        7.191 |                    nan |   6.944 |             1     |
|      3 |           64 |     0.5 | on         | on        |              4 |           3 |       7.023 |       3.501 |        7.191 |                      1 |   6.944 |             0.667 |
|      3 |           64 |     0.5 | on         | on        |              4 |           4 |       6.944 |       3.501 |        7.191 |                      1 |   6.944 |             0.5   |
|      4 |           64 |     0.5 | on         | on        |              4 |           1 |       7.139 |       4.501 |        6.901 |                      0 |   6.914 |             2     |
|      4 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       4.501 |        6.901 |                    nan |   6.914 |             1     |
|      4 |           64 |     0.5 | on         | on        |              4 |           3 |       6.947 |       4.501 |        6.901 |                      0 |   6.914 |             0.667 |
|      4 |           64 |     0.5 | on         | on        |              4 |           4 |       6.914 |       4.501 |        6.901 |                      0 |   6.914 |             0.5   |
