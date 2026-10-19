Worst-case perplexity with *s* of *N* nodes alive. *replicated*: every node holds the same round-to-nearest model (same memory as a child); `replicated_match_bpw` is the extra memory per node at which it becomes at least as good as staggered. *coded*: every node holds one base child plus an erasure-coded share of the parent's remaining bits, any *s* of which rebuild the parent; `coded_extra_bpw` is its extra memory per node. Quantized weights are 100.0% of all parameters. Blank staggered cells were not measured.

|   bits |   group_size |   phase | rotation   | scaling   |   num_children |   survivors |   staggered |   child_bpw |   replicated |   replicated_match_bpw |   coded |   coded_extra_bpw |
|-------:|-------------:|--------:|:-----------|:----------|---------------:|------------:|------------:|------------:|-------------:|-----------------------:|--------:|------------------:|
|      4 |           32 |     0.5 | off        | off       |              4 |           1 |      10.754 |       5.003 |       10.673 |                  0     |  10.09  |             1.999 |
|      4 |           32 |     0.5 | off        | off       |              4 |           2 |      10.341 |       5.003 |       10.673 |                  1     |  10.09  |             1     |
|      4 |           32 |     0.5 | off        | off       |              4 |           3 |      10.156 |       5.003 |       10.673 |                  1.999 |  10.09  |             0.666 |
|      4 |           32 |     0.5 | off        | off       |              4 |           4 |      10.09  |       5.003 |       10.673 |                  1.999 |  10.09  |             0.5   |
|      4 |           32 |     0.5 | off        | on        |              4 |           1 |      10.531 |       5.003 |       10.439 |                  0     |  10.072 |             1.999 |
|      4 |           32 |     0.5 | off        | on        |              4 |           2 |      10.247 |       5.003 |       10.439 |                  1     |  10.072 |             1     |
|      4 |           32 |     0.5 | off        | on        |              4 |           3 |      10.114 |       5.003 |       10.439 |                  1.999 |  10.072 |             0.666 |
|      4 |           32 |     0.5 | off        | on        |              4 |           4 |      10.072 |       5.003 |       10.439 |                  3.999 |  10.072 |             0.5   |
|      4 |           32 |     0.5 | on         | off       |              4 |           1 |      10.699 |       5.003 |       10.551 |                  0     |  10.087 |             1.999 |
|      4 |           32 |     0.5 | on         | off       |              4 |           2 |      10.329 |       5.003 |       10.551 |                  1     |  10.087 |             1     |
|      4 |           32 |     0.5 | on         | off       |              4 |           3 |      10.145 |       5.003 |       10.551 |                  1.999 |  10.087 |             0.666 |
|      4 |           32 |     0.5 | on         | off       |              4 |           4 |      10.087 |       5.003 |       10.551 |                  1.999 |  10.087 |             0.5   |
|      4 |           32 |     0.5 | on         | on        |              4 |           1 |      10.473 |       5.003 |       10.417 |                  0     |  10.077 |             1.999 |
|      4 |           32 |     0.5 | on         | on        |              4 |           2 |      10.243 |       5.003 |       10.417 |                  1     |  10.077 |             1     |
|      4 |           32 |     0.5 | on         | on        |              4 |           3 |      10.119 |       5.003 |       10.417 |                  1.999 |  10.077 |             0.666 |
|      4 |           32 |     0.5 | on         | on        |              4 |           4 |      10.077 |       5.003 |       10.417 |                  1.999 |  10.077 |             0.5   |
