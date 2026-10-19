Worst-case perplexity with *s* of *N* nodes alive. *replicated*: every node holds the same round-to-nearest model (same memory as a child); `replicated_match_bpw` is the extra memory per node at which it becomes at least as good as staggered. *coded*: every node holds one base child plus an erasure-coded share of the parent's remaining bits, any *s* of which rebuild the parent; `coded_extra_bpw` is its extra memory per node. Quantized weights are 100.0% of all parameters. Blank staggered cells were not measured.

|   bits |   group_size |   phase | rotation   | scaling   |   num_children |   survivors |   staggered |   child_bpw |   replicated |   replicated_match_bpw |   coded |   coded_extra_bpw |
|-------:|-------------:|--------:|:-----------|:----------|---------------:|------------:|------------:|------------:|-------------:|-----------------------:|--------:|------------------:|
|      2 |           64 |     0.5 | on         | on        |              4 |           1 |      36.542 |       2.502 |       31.265 |                      0 |   8.788 |             2     |
|      2 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       2.502 |       31.265 |                    nan |   8.788 |             1     |
|      2 |           64 |     0.5 | on         | on        |              4 |           3 |       9.328 |       2.502 |       31.265 |                      1 |   8.788 |             0.667 |
|      2 |           64 |     0.5 | on         | on        |              4 |           4 |       8.788 |       2.502 |       31.265 |                    nan |   8.788 |             0.5   |
|      3 |           64 |     0.5 | on         | on        |              4 |           1 |       9.836 |       3.501 |        9.207 |                      0 |   8.645 |             2     |
|      3 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       3.501 |        9.207 |                    nan |   8.645 |             1     |
|      3 |           64 |     0.5 | on         | on        |              4 |           3 |       8.847 |       3.501 |        9.207 |                      1 |   8.645 |             0.667 |
|      3 |           64 |     0.5 | on         | on        |              4 |           4 |       8.645 |       3.501 |        9.207 |                    nan |   8.645 |             0.5   |
|      4 |           64 |     0.5 | on         | on        |              4 |           1 |       9.07  |       4.501 |        8.835 |                      0 |   8.675 |             2     |
|      4 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       4.501 |        8.835 |                    nan |   8.675 |             1     |
|      4 |           64 |     0.5 | on         | on        |              4 |           3 |       8.755 |       4.501 |        8.835 |                    nan |   8.675 |             0.667 |
|      4 |           64 |     0.5 | on         | on        |              4 |           4 |       8.675 |       4.501 |        8.835 |                    nan |   8.675 |             0.5   |
