bf16 perplexity: **10.052** · full pool: **10.195** · single child: **12.862** (mean of 4)

5 stages x 4 devices. The largest device holds 0.222 GB of weights; the bf16 model is 3.764 GB (16.9x larger).

## One device lost (perplexity by stage)

|   stage |   mean (avg) |   midpoint (avg) |   mean (worst) |   midpoint (worst) |
|--------:|-------------:|-----------------:|---------------:|-------------------:|
|       0 |       10.255 |           10.249 |         10.274 |             10.268 |
|       1 |       10.243 |           10.239 |         10.253 |             10.246 |
|       2 |       10.226 |           10.223 |         10.228 |             10.225 |
|       3 |       10.232 |           10.228 |         10.237 |             10.233 |
|       4 |       10.249 |           10.244 |         10.259 |             10.251 |

## One stage on a single child

|   stage |    avg |   worst |
|--------:|-------:|--------:|
|       0 | 10.753 |  10.791 |
|       1 | 10.707 |  10.752 |
|       2 | 10.5   |  10.52  |
|       3 | 10.543 |  10.549 |
|       4 | 10.726 |  10.767 |

## Random device losses

|                 |   count |    avg |   worst |
|:----------------|--------:|-------:|--------:|
| ('mean', 2)     |       8 | 10.285 |  10.334 |
| ('mean', 3)     |       8 | 10.37  |  10.474 |
| ('mean', 4)     |       8 | 10.423 |  10.501 |
| ('mean', 5)     |       8 | 10.501 |  10.569 |
| ('mean', 6)     |       8 | 10.77  |  11.027 |
| ('midpoint', 2) |       8 | 10.278 |  10.334 |
| ('midpoint', 3) |       8 | 10.361 |  10.472 |
| ('midpoint', 4) |       8 | 10.409 |  10.489 |
| ('midpoint', 5) |       8 | 10.487 |  10.558 |
| ('midpoint', 6) |       8 | 10.762 |  11.025 |

## Additivity across stages (log perplexity)

| estimator   |   patterns |   r_squared |   measured / predicted |
|:------------|-----------:|------------:|-----------------------:|
| mean        |         40 |       0.999 |                  1.003 |
| midpoint    |         40 |       0.999 |                  1.003 |

## Hypothesis checks

- H6b, one lost device stays within 0.5 of bf16: worst gap 0.222, **pass**
- H6c, stage-wise prediction R² ≥ 0.9 (mean): 0.999, **pass**
- H6c, stage-wise prediction R² ≥ 0.9 (midpoint): 0.999, **pass**
