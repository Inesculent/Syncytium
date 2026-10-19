Worst-case perplexity with *s* of *N* nodes alive. *replicated*: every node holds the same round-to-nearest model (same memory as a child); `replicated_match_bpw` is the extra memory per node at which it becomes at least as good as staggered. *coded*: every node holds one base child plus an erasure-coded share of the parent's remaining bits, any *s* of which rebuild the parent; `coded_extra_bpw` is its extra memory per node. Quantized weights are 100.0% of all parameters. Blank staggered cells were not measured.

|   bits |   group_size |   phase | rotation   | scaling   |   num_children |   survivors |   staggered |   child_bpw |   replicated |   replicated_match_bpw |   coded |   coded_extra_bpw |
|-------:|-------------:|--------:|:-----------|:----------|---------------:|------------:|------------:|------------:|-------------:|-----------------------:|--------:|------------------:|
|      2 |           64 |     0.5 | on         | on        |              4 |           1 |    1015.91  |       2.504 |      375.849 |                  0     |  13.353 |             1.999 |
|      2 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       2.504 |      375.849 |                nan     |  13.353 |             1     |
|      2 |           64 |     0.5 | on         | on        |              4 |           3 |      15.603 |       2.504 |      375.849 |                  1     |  13.353 |             0.666 |
|      2 |           64 |     0.5 | on         | on        |              4 |           4 |      13.353 |       2.504 |      375.849 |                  1.999 |  13.353 |             0.5   |
|      3 |           64 |     0.5 | on         | on        |              4 |           1 |      16.047 |       3.504 |       15.453 |                  0     |  12.276 |             1.999 |
|      3 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       3.504 |       15.453 |                nan     |  12.276 |             1     |
|      3 |           64 |     0.5 | on         | on        |              4 |           3 |      12.691 |       3.504 |       15.453 |                  1     |  12.276 |             0.666 |
|      3 |           64 |     0.5 | on         | on        |              4 |           4 |      12.276 |       3.504 |       15.453 |                nan     |  12.276 |             0.5   |
|      4 |           64 |     0.5 | on         | on        |              4 |           1 |      13.015 |       4.503 |       12.62  |                  0     |  12.188 |             1.999 |
|      4 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       4.503 |       12.62  |                nan     |  12.188 |             1     |
|      4 |           64 |     0.5 | on         | on        |              4 |           3 |      12.347 |       4.503 |       12.62  |                nan     |  12.188 |             0.666 |
|      4 |           64 |     0.5 | on         | on        |              4 |           4 |      12.188 |       4.503 |       12.62  |                nan     |  12.188 |             0.5   |
