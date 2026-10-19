bf16 perplexity: **13.125** · full pool: **13.197** · single child: **14.712** (mean of 4)

5 stages x 4 devices. The largest device holds 0.143 GB of weights; the bf16 model is 1.505 GB (10.5x larger).

## One device lost (perplexity by stage)

|   stage |   mean (avg) |   midpoint (avg) |   mean (worst) |   midpoint (worst) |
|--------:|-------------:|-----------------:|---------------:|-------------------:|
|       0 |       13.225 |           13.221 |         13.234 |             13.233 |
|       1 |       13.237 |           13.234 |         13.246 |             13.249 |
|       2 |       13.216 |           13.214 |         13.223 |             13.221 |
|       3 |       13.22  |           13.218 |         13.226 |             13.225 |
|       4 |       13.232 |           13.229 |         13.234 |             13.232 |

## One stage on a single child

|   stage |    avg |   worst |
|--------:|-------:|--------:|
|       0 | 13.452 |  13.477 |
|       1 | 13.601 |  13.655 |
|       2 | 13.376 |  13.403 |
|       3 | 13.414 |  13.436 |
|       4 | 13.532 |  13.552 |

## Random device losses

|                 |   count |    avg |   worst |
|:----------------|--------:|-------:|--------:|
| ('mean', 2)     |       8 | 13.261 |  13.291 |
| ('mean', 3)     |       8 | 13.301 |  13.327 |
| ('mean', 4)     |       8 | 13.344 |  13.407 |
| ('mean', 5)     |       8 | 13.381 |  13.427 |
| ('mean', 6)     |       8 | 13.512 |  13.631 |
| ('midpoint', 2) |       8 | 13.258 |  13.29  |
| ('midpoint', 3) |       8 | 13.297 |  13.323 |
| ('midpoint', 4) |       8 | 13.336 |  13.404 |
| ('midpoint', 5) |       8 | 13.373 |  13.427 |
| ('midpoint', 6) |       8 | 13.506 |  13.629 |

## Additivity across stages (log perplexity)

| estimator   |   patterns |   r_squared |   measured / predicted |
|:------------|-----------:|------------:|-----------------------:|
| mean        |         40 |           1 |                  1.005 |
| midpoint    |         40 |           1 |                  1.003 |

## Hypothesis checks

- H6b, one lost device stays within 0.5 of bf16: worst gap 0.124, **pass**
- H6c, stage-wise prediction R² ≥ 0.9 (mean): 1.000, **pass**
- H6c, stage-wise prediction R² ≥ 0.9 (midpoint): 1.000, **pass**
