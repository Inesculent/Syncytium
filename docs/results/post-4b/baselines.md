Worst-case perplexity with *s* of *N* nodes alive. *replicated*: every node holds the same round-to-nearest model (same memory as a child); `replicated_match_bpw` is the extra memory per node at which it becomes at least as good as staggered. *coded*: every node holds one base child plus an erasure-coded share of the parent's remaining bits, any *s* of which rebuild the parent; `coded_extra_bpw` is its extra memory per node. Quantized weights are 100.0% of all parameters. Blank staggered cells were not measured.

|   bits |   group_size |   phase | rotation   | scaling   |   num_children |   survivors |   staggered |   child_bpw |   replicated |   replicated_match_bpw |   coded |   coded_extra_bpw |
|-------:|-------------:|--------:|:-----------|:----------|---------------:|------------:|------------:|------------:|-------------:|-----------------------:|--------:|------------------:|
|      2 |           64 |     0.5 | on         | on        |              4 |           1 |     173.203 |       2.503 |      113.455 |                      0 |  10.312 |             2     |
|      2 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       2.503 |      113.455 |                    nan |  10.312 |             1     |
|      2 |           64 |     0.5 | on         | on        |              4 |           3 |      11.448 |       2.503 |      113.455 |                      2 |  10.312 |             0.667 |
|      2 |           64 |     0.5 | on         | on        |              4 |           4 |      10.312 |       2.503 |      113.455 |                      2 |  10.312 |             0.5   |
|      3 |           64 |     0.5 | on         | on        |              4 |           1 |      11.7   |       3.503 |       11.719 |                      1 |   9.732 |             2     |
|      3 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       3.503 |       11.719 |                    nan |   9.732 |             1     |
|      3 |           64 |     0.5 | on         | on        |              4 |           3 |       9.939 |       3.503 |       11.719 |                      1 |   9.732 |             0.667 |
|      3 |           64 |     0.5 | on         | on        |              4 |           4 |       9.732 |       3.503 |       11.719 |                    nan |   9.732 |             0.5   |
|      4 |           64 |     0.5 | on         | on        |              4 |           1 |      10.105 |       4.503 |        9.887 |                      0 |   9.588 |             2     |
|      4 |           64 |     0.5 | on         | on        |              4 |           2 |     nan     |       4.503 |        9.887 |                    nan |   9.588 |             1     |
|      4 |           64 |     0.5 | on         | on        |              4 |           3 |       9.693 |       4.503 |        9.887 |                    nan |   9.588 |             0.667 |
|      4 |           64 |     0.5 | on         | on        |              4 |           4 |       9.588 |       4.503 |        9.887 |                    nan |   9.588 |             0.5   |
