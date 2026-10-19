Worst-case perplexity with *s* of *N* nodes alive. *replicated*: every node holds the same round-to-nearest model (same memory as a child); `replicated_match_bpw` is the extra memory per node at which it becomes at least as good as staggered. *coded*: every node holds one base child plus an erasure-coded share of the parent's remaining bits, any *s* of which rebuild the parent; `coded_extra_bpw` is its extra memory per node. Quantized weights are 99.9% of all parameters. Blank staggered cells were not measured.

|   bits |   group_size |   phase | rotation   | scaling   |   num_children |   survivors |   staggered |   child_bpw |   replicated |   replicated_match_bpw |   coded |   coded_extra_bpw |
|-------:|-------------:|--------:|:-----------|:----------|---------------:|------------:|------------:|------------:|-------------:|-----------------------:|--------:|------------------:|
|      2 |           64 |     0.5 | on         | on        |              4 |           1 |    7161.81  |       2.509 |     3752.95  |                  0     |  19.86  |             1.999 |
|      2 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       2.509 |     3752.95  |                nan     |  19.86  |             0.999 |
|      2 |           64 |     0.5 | on         | on        |              4 |           3 |      25.649 |       2.509 |     3752.95  |                  1.999 |  19.86  |             0.666 |
|      2 |           64 |     0.5 | on         | on        |              4 |           4 |      19.86  |       2.509 |     3752.95  |                  1.999 |  19.86  |             0.5   |
|      3 |           64 |     0.5 | on         | on        |              4 |           1 |      27.544 |       3.508 |       26.622 |                  0     |  17.232 |             1.999 |
|      3 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       3.508 |       26.622 |                nan     |  17.232 |             0.999 |
|      3 |           64 |     0.5 | on         | on        |              4 |           3 |      18.172 |       3.508 |       26.622 |                nan     |  17.232 |             0.666 |
|      3 |           64 |     0.5 | on         | on        |              4 |           4 |      17.232 |       3.508 |       26.622 |                nan     |  17.232 |             0.5   |
|      4 |           64 |     0.5 | on         | on        |              4 |           1 |      19.14  |       4.508 |       18.217 |                  0     |  17.292 |             1.999 |
|      4 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       4.508 |       18.217 |                nan     |  17.292 |             0.999 |
|      4 |           64 |     0.5 | on         | on        |              4 |           3 |      17.518 |       4.508 |       18.217 |                nan     |  17.292 |             0.666 |
|      4 |           64 |     0.5 | on         | on        |              4 |           4 |      17.292 |       4.508 |       18.217 |                nan     |  17.292 |             0.5   |
