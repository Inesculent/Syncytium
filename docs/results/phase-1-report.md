# Phase 1 Report: Staggered Children at 0.8B, 2B, 4B and 9B

*Draft, 2026-10-18. Covers the 0.8B pilot, the 2B, 4B and 9B sweeps, every survivor count for
N = 4 at 0.8B and 4B (`subsets-0.8b`, `subsets-4b`), activation-aware channel scaling at every
size (stage 1.9), rotation seeds at 9B, the same-node baselines, the equal-memory comparison
across model sizes, the demo's scheme at 2B and the pooled pipeline at 0.8B and 2B. After
phase 1 the project was reframed as a swarm of agents; its results are in the
[swarm report](swarm-report.md), and the "Implications for phase 2" below predate it.*

*For the whole project in one place, with conclusions and limitations, see the
[research report](../report.md).*

## Summary

- **The scheme works as designed, and it gets better as the model grows.** At 4B and 9B,
  every 3- and 4-bit child is a usable model on its own (worst child at most 1.41x bf16
  perplexity), and four children rebuild a parent within 0.03 (4-bit) or 0.13 (3-bit) of
  bf16. 4-bit children are usable at every size; 3-bit children cross the usability bar
  between 2B and 4B, or at 2B with channel scaling.
- **A better quantizer stacks on top instead of replacing it.** AWQ-style channel scaling
  keeps the rebuild exact and improves every child, rebuild and single model. With it, the
  worst 3-bit child of the 9B model drops from 1.33x to 1.13x bf16, and four 2-bit children
  rebuild the 4B and 9B models within H3's 0.5 bar (0.44 and 0.27).
- **With scaling and rotation, the last node standing pays little.** Against the scaled,
  rotated RTN model of the same size, the worst staggered child at 9B is behind by at most
  0.014 at 4 bits across three rotation seeds (7.159 against 7.157 with seed 0), and by
  0.05–0.09 at 3 bits. Without scaling it was behind by up to 0.49 at 3 bits; with scaling
  but no rotation, by about 0.2. *(Corrected after the seed runs: the draft claimed at most
  0.05 at 3 bits as well, which held only for seed 0.)*
- **Quality degrades strictly with each lost child.** At 0.8B and 4B with four children, the
  worst rebuild from any *k* + 1 children beats the best from any *k*, for every *k* and
  every setting. Which children survive also matters: two children with opposite offsets rebuild
  far better than two adjacent ones.
- **A fleet of small nodes beats any single model that fits on one of them.** At the memory
  of one child, any three of four scaled children rebuild a better model than the best
  single model (scaled or not) of any measured size that fits, in every setting at every
  size, 2-bit included. Four scaled 2-bit 9B children of 2.8 GB rebuild to 7.25, against 8.34
  for the best model that fits in 2.8 GB.
- **For phase 2:** channel scaling is the largest cheap gain available to the demo, but
  shards cannot store the scales yet and llama.cpp needs them folded into the weights. 2B is
  the natural demo model for local hardware. The two-of-four results give the orchestrator
  a concrete rule for placing children across failure domains; see
  [implications for phase 2](#implications-for-phase-2-the-orchestrated-system).

## What was run

| Item | Setting |
|------|---------|
| Models | Qwen3.5 Base at 0.8B (0.75B parameters), 2B (1.88B), 4B (4.21B) and 9B (8.95B) |
| Metric | WikiText-2 test perplexity, non-overlapping 2048-token windows over the whole split |
| Grid | *b* ∈ {2, 3, 4}, *N* ∈ {2, 4, 8}, *g* = 64, phase φ ∈ {0, ½}, rotation off/on (seed 0) |
| Conditions per setting | bf16; every child alone; full rebuild; every rebuild with one child missing; RTN at 2–8 bits |
| Survivor counts | `subsets-0.8b`: *N* = 4 only, every rebuild with one or two children missing |
| Channel scaling | `scaling-*`: *N* = 4, φ = ½, rotation off/on, scaling off/on, one child missing; scaled and unscaled RTN at 2–8 bits |
| Hardware | RTX 3080 (0.8B pilot), H100 PCIe (everything else); about 2.6–3.1 GPU-hours per sweep |
| Totals | 387 logged conditions per sweep, 1,548 across the four, plus 195 in `subsets-0.8b` and 548 in the scaling runs |

Phase 1 swaps dequantized weights into a bf16 model, so it measures quality and memory, not
speed. Memory figures count codes, fp16 scales and every unquantized tensor; channel scales
add under 0.006 bits per weight and are not counted ([method](../method.md#optional-channel-scaling)).
Channel scales are calibrated on the WikiText-2 *train* split, never the test split.
Rebuilds with children missing use the interval-midpoint estimator
([method](../method.md#partial-rebuild)).

Reference perplexities (bf16): **0.8B 13.125, 2B 10.052, 4B 8.055, 9B 6.982**.

**Reproducibility.** `subsets-0.8b` repeats 123 of the pilot's conditions on a different GPU
(H100 against RTX 3080). Every perplexity below 100 agrees within 0.005 (bf16: 13.1248
against 13.1254). The scaling runs repeat 69 unscaled conditions of each sweep, which agree
exactly on the same GPU type. Measurement noise is therefore far below every difference this
report draws a conclusion from; the remaining uncertainty is from using one rotation seed.

## Results by hypothesis

Thresholds were fixed after the 0.8B pilot and before any 2B, 4B or 9B result was examined
([hypothesis](../hypothesis.md#what-results-would-mean)): H2 needs the worst child within
1.5x of bf16; H3 needs the full rebuild within 0.5 of bf16; H4 is checked for *N* ≥ 3.
The tables in this section are unscaled; [channel scaling](#channel-scaling-stage-19) has
its own section.

### H1: exact superposition: holds

The rebuild identity is covered by unit tests (with and without rotation, both phases), and
the shard writer of stage 2.1 rebuilds a 0.8B set from packed files with a matching SHA-256
of the parent's codes. This is an implementation check; the identity itself is Hermite's.

### H2: functional children: holds at 4 bits everywhere, at 3 bits from 4B

Worst child as a multiple of bf16 perplexity, over the four (φ, rotation) settings:

| Bits | Memory vs bf16 | 0.8B | 2B | 4B | 9B |
|-----:|---------------:|-----:|---:|---:|---:|
| 4 | 3.6x smaller | 1.12–1.17 ✅ | 1.08–1.10 ✅ | 1.05–1.06 ✅ | 1.05–1.06 ✅ |
| 3 | 4.6x smaller | 1.97–2.70 ❌ | 1.54–1.78 ❌ | 1.31–1.40 ✅ | 1.32–1.41 ✅ |
| 2 | 6.4x smaller | > 2,000 ❌ | > 700 ❌ | > 400 ❌ | > 400 ❌ |

The ratio falls steadily from 0.8B to 4B and then levels off: 3-bit children improve from
2.0–2.7x to 1.5–1.8x to 1.3–1.4x, and do not improve further at 9B. The best 2B setting
(φ = ½ with rotation, 1.54x) misses the bar narrowly; with channel scaling it passes
(1.30x). "Bigger is better" should not be extrapolated past 9B without more sizes.

### H2a: rotation helps low-bit children: holds for children, not rebuilds

The shared Hadamard rotation lowers the worst 3-bit child at every size (0.8B 32.0 → 27.2,
2B 17.46 → 15.69, 4B 11.07 → 10.82, 9B 9.61 → 9.27, all at φ = ½) and the worst 2-bit child
by up to 8x (still unusable). At 4 bits it moves the worst child by 0.03–0.05 at 4B and 9B,
0.13–0.17 at 2B and 0.3–0.5 at 0.8B. Full rebuilds move by small amounts in either
direction. At 4B and 9B, 3-bit children pass H2 *without* rotation, so rotation is a
refinement there rather than an enabler. φ = ½ with rotation gives the best worst-child in
most settings. Rotation and channel scaling combine: the best setting in every scaling run
uses both.

### H3: near-lossless rebuild: holds whenever the rebuild has about 5+ bits

Full-rebuild gap to bf16 (perplexity), φ = ½ with rotation:

| Bits | *N* | Parent bits | 0.8B | 2B | 4B | 9B |
|-----:|----:|------------:|-----:|---:|---:|---:|
| 4 | 2 | 5 | 0.35 ✅ | 0.19 ✅ | 0.09 ✅ | 0.08 ✅ |
| 4 | 4 | 6 | 0.07 ✅ | 0.04 ✅ | 0.02 ✅ | 0.02 ✅ |
| 4 | 8 | 7 | 0.03 ✅ | 0.02 ✅ | 0.01 ✅ | 0.00 ✅ |
| 3 | 2 | 4 | 1.74 ❌ | 0.78 ❌ | 0.43 ✅ | 0.38 ✅ |
| 3 | 4 | 5 | 0.39 ✅ | 0.18 ✅ | 0.10 ✅ | 0.10 ✅ |
| 3 | 8 | 6 | 0.09 ✅ | 0.06 ✅ | 0.03 ✅ | 0.02 ✅ |
| 2 | 4 | 4 | 2.77 ❌ | 1.18 ❌ | 0.65 ❌ | 0.58 ❌ |
| 2 | 8 | 5 | 0.54 ❌ | 0.31 ✅ | 0.14 ✅ | 0.11 ✅ |

Across all settings, every 4-bit setting passes. 3-bit passes for *N* ≥ 4 at every size
(except one unrotated φ = 0 setting at 0.8B), and also for *N* = 2 at 9B and partly at 4B.
Unscaled, 2-bit passes only with eight children, from 2B up; with channel scaling, four
2-bit children also pass at 4B and 9B. The gap shrinks by roughly 4x or more each time *N*
doubles until it reaches a floor of a few thousandths, consistent with the rebuild's error
falling as 1/*N* ([method](../method.md#rebuild)) and the perplexity gap tracking squared
error.

### H4: graceful degradation: holds, including with two children missing

**One missing child, every size.** For every *N* ≥ 4 setting at every size, scaled or not,
the worst rebuild from *N* − 1 children beats the best single child. At 9B with φ = ½ and
rotation, losing one of four children costs 0.04 (4-bit), 0.17 (3-bit) and 1.2 (2-bit)
perplexity; losing one of eight costs 0.01, 0.02 and 0.12.

**Every survivor count, 0.8B, *N* = 4** (`subsets-0.8b`). Range over every subset of each
size, φ = ½ with rotation:

| Children alive | 4-bit | 3-bit | 2-bit |
|---------------:|------:|------:|------:|
| 4 | 13.20 | 13.52 | 15.90 |
| 3 | 13.32–13.34 | 14.16–14.27 | 22.4–23.2 |
| 2 | 13.46–13.77 | 14.90–16.96 | 38–262 |
| 1 | 14.61–14.85 | 24.5–27.2 | 18,600–43,100 |

In all twelve (bits, φ, rotation) settings, the ranges do not overlap: **any *k* + 1
children rebuild a better model than any *k***. This is stronger than H4 asks.

**Which two survive matters.** The two-child rebuilds split cleanly by offset spacing:

| 0.8B, φ = ½, rotation | Opposite offsets (½ apart) | Adjacent offsets (¼ apart) |
|-----------------------|------------------:|------------------:|
| 4-bit | 13.46–13.49 | 13.71–13.77 |
| 3-bit | 14.90–15.02 | 16.47–16.96 |
| 2-bit | 38–40 | 155–262 |

This is what the partial-rebuild bound predicts: the error is half the largest gap between
the surviving offsets, Δ/4 for an opposite pair against 3Δ/8 for an adjacent one. An
opposite pair is about as good as a full *N* = 2 set (3-bit: 14.86).

**At 4B** (`subsets-4b`, φ = ½, with and without rotation and scaling, every survivor count),
both pre-registered criteria hold in all twelve settings: any *k* + 1 children beat any *k*,
and every opposite pair beats every adjacent pair. 3-bit, rotated and scaled: one child
9.62–9.67, two 8.39–8.68 (opposite 8.39–8.40, adjacent 8.63–8.68), three 8.25–8.27, all four
8.14. The placement rule below therefore holds where 3-bit children are usable, not only at
0.8B.

### H5: survival in a running system: superseded

The reframed project has no main container to wipe and rebuild. H5 is replaced by H13–H15 of
the [swarm pre-registration](../preregistration-swarm.md): a fused group surviving a member's
loss mid-answer, and exact regrowth of lost members (see the [swarm report](swarm-report.md)).

### H6: pooled pipeline: holds at 0.8B and 2B

Five stages of four devices each, every single-device loss, every stage on one child and 40
random loss patterns of 2–6 devices ([pool-0.8b](pool-0.8b/summary.md),
[pool-2b](pool-2b/summary.md)):

| | 0.8B, 4-bit, rotated | 2B, 3-bit, rotated and scaled |
|---|---:|---:|
| bf16 / full pool | 13.125 / 13.197 | 10.052 / 10.195 |
| H6b: worst single device lost, gap to bf16 (bar 0.5) | 0.124 ✅ | 0.222 ✅ |
| H6c: R² of the stage-wise prediction, mean estimator (bar 0.9) | 1.000 ✅ | 0.999 ✅ |
| Measured / predicted log-perplexity increase | 1.005 | 1.003 |

The cost of any loss pattern is the sum of each stage's own cost, almost exactly, under
both estimators. The midpoint estimator beats the mean by only 0.003 (0.8B) and 0.005 (2B)
with one device lost. (The earlier `pool-screen-0.8b` check had 3 stages, not 2 as first
written.)

## Channel scaling (stage 1.9)

Activation-aware channel scales, shared by every child, are the strongest quantizer
improvement that keeps the rebuild exact ([method](../method.md#optional-channel-scaling));
GPTQ-style error feedback does not compose with staggering. Measured at *N* = 4, φ = ½, with
and without rotation, at every size. Rotated settings, unscaled → scaled:

| | 0.8B | 2B | 4B | 9B |
|---|-----:|---:|---:|---:|
| Worst 4-bit child (x bf16) | 1.13 → 1.09 ✅ | 1.08 → 1.05 ✅ | 1.05 → 1.04 ✅ | 1.05 → 1.03 ✅ |
| Worst 3-bit child (x bf16) | 2.07 → 1.59 ❌ | 1.56 → **1.30** ✅ | 1.34 → 1.20 ✅ | 1.33 → 1.13 ✅ |
| Worst 2-bit child (x bf16) | 3,282 → 391 ❌ | 1,216 → 46 ❌ | 530 → 12 ❌ | 891 → 3.7 ❌ |
| 4-bit rebuild gap | 0.07 → 0.05 | 0.04 → 0.03 | 0.02 → 0.02 | 0.02 → 0.01 |
| 3-bit rebuild gap | 0.39 → 0.29 | 0.18 → 0.14 | 0.10 → 0.08 | 0.10 → 0.05 |
| 2-bit rebuild gap | 2.77 → 1.66 ❌ | 1.18 → 0.79 ❌ | 0.65 → **0.44** ✅ | 0.58 → **0.27** ✅ |

Scaling helps children far more than rebuilds, and helps most at low bit widths. It
changes three verdicts: 2B's 3-bit children now pass H2 (1.30x rotated, 1.40x unrotated),
and four 2-bit children pass H3 at 4B (0.44–0.49) and 9B (0.27–0.30), with or without
rotation. 2-bit children are still far from usable alone, but at 9B with rotation they
move from thousands to 21–26.

**Scaling removes most of the single-node penalty.** Staggered children used to trail the
RTN model of the same bit width partly because their offsets placed levels unevenly. With
scaling, the four children of a set become nearly interchangeable (9B 3-bit: 7.86–7.90,
against 8.88–9.27 unscaled), and the worst one is almost as good as the scaled RTN model:

| Rotated, φ = ½ | Worst child, unscaled | RTN, unscaled | Worst child, scaled | RTN, scaled |
|----------------|------:|------:|------:|------:|
| 4B, 3-bit | 10.82 | 10.33 | 9.668 | 9.662 |
| 4B, 4-bit | 8.465 | 8.471 | 8.365 | 8.335 |
| 9B, 3-bit | 9.27 | 8.92 | 7.903 | 7.854 |
| 9B, 4-bit | 7.335 | 7.298 | 7.159 | 7.157 |

At 2B the scaled gap is still 0.39 (3-bit) and 0.06 (4-bit), and at 0.8B 1.0 and 0.09.
Rotation matters here: scaled but unrotated 3-bit children trail the scaled, unrotated RTN
model by about 0.21 at both 4B and 9B, and their spread is 0.2 rather than 0.05.

**Across rotation seeds** (`seed1-9b`, `seed2-9b`), the 9B 4-bit gap stays at most 0.014,
but the 3-bit gap is 0.048, 0.055 and 0.093 for seeds 0, 1 and 2. The pre-registered
criterion (within 0.05 for every seed at 3 and 4 bits) therefore fails at 3 bits. Scaled
rebuilds move by at most 0.006 across seeds, single children by about 0.02 and RTN models by
about 0.03, so differences between single 9B models below about 0.03 cannot be ranked.

**The answer to "why not a better quantizer?"** is that the two are complementary: scaling
improves the quantizer, and staggering then turns *N* copies of it into functional shards of
a higher-precision parent at no extra memory. Whatever scaling buys a single model, the
staggered set gets as well.

## Comparison with simpler schemes

### Same nodes, same memory

Worst-case perplexity with *s* of 4 nodes alive, 9B, φ = ½ with rotation
([unscaled](sweep-9b/baselines.md), [scaled](scaling-9b/baselines.md)):

| Scheme | Extra memory / node | 3-bit, *s* = 1 | 3-bit, *s* = 3 | 3-bit, *s* = 4 | 4-bit, *s* = 1 | 4-bit, *s* = 3 | 4-bit, *s* = 4 |
|--------|--------------------:|------:|------:|------:|------:|------:|------:|
| Staggered children | 0 | 9.27 | **7.25** | **7.08** | 7.34 | **7.04** | **7.00** |
| Same RTN model on every node | 0 | **8.92** | 8.92 | 8.92 | **7.30** | 7.30 | 7.30 |
| Base child + coded remainder, *k* = 3 | +0.67 bpw | 9.11 | 7.08 | 7.08 | 7.32 | 7.00 | 7.00 |
| Staggered children, scaled | 0 | 7.90 | **7.11** | **7.03** | 7.16 | **7.01** | **6.99** |
| Same scaled RTN model on every node | 0 | **7.85** | 7.85 | 7.85 | **7.16** | 7.16 | 7.16 |

Two survivors were measured only at 0.8B and unscaled ([table](subsets-0.8b/baselines.md)).
There, with two of four alive, staggered children score 16.96 (3-bit) and 13.77 (4-bit) in
the worst case, against 25.45 and 14.66 for replication; replication needs one more bit per
node to match. A coded remainder with *k* = 2 rebuilds the parent (13.52 and 13.20) for one
extra bit per node, and one with *k* = 3 falls back to its base child.

- **Replication** wins only with one node left, and with scaling only by 0.05 or less at
  4B and 9B. With two of four it needs about 1 more bit per node to match staggered
  children, and with three of four about 2 more.
- **The coded remainder** rebuilds the exact parent from any *k* nodes for log₂(*N*)/*k*
  extra bits per node, and serves the base child below *k*. It beats staggered at *s* ≥ *k*
  if that memory is available; with all four alive, staggered matches it at no extra memory.
- **Staggered children are the only scheme whose quality rises with every surviving node at
  zero extra memory**, and the only one whose shards can cooperate without first being
  decoded (pooling). All four sizes show the same pattern, scaled or not.

### Equal memory, any model size

At the weight memory of one child, compared with the best single model of any measured size
that fits: bf16, or RTN at 2–8 bits with or without rotation and scaling
([scaled table](sizes-scaled/sizes.md), [figure](sizes-scaled/sizes-p0.5-rot-on-scaled.png);
unscaled-only comparison in [sizes](sizes/sizes.md)). *N* = 4, φ = ½, rotation and scaling:

| Setting | Memory / node | One node alive | Any 3 of 4 | All 4 | Best single fit |
|---------|--------------:|---------------:|-----------:|------:|----------------:|
| 2B, 3-bit | 0.82 GB | 13.02 | 10.43 | 10.20 | 12.64 (2B 3-bit) |
| 2B, 4-bit | 1.06 GB | 10.58 | 10.13 | 10.08 | 10.52 (2B 4-bit) |
| 4B, 2-bit | 1.32 GB | 96.9 | 9.39 | 8.49 | 10.17 (2B 5-bit) |
| 4B, 3-bit | 1.84 GB | 9.67 | 8.27 | 8.14 | 9.66 (4B 3-bit) |
| 4B, 4-bit | 2.37 GB | 8.36 | 8.10 | 8.07 | 8.34 (4B 4-bit) |
| 9B, 2-bit | 2.80 GB | 25.6 | 7.75 | 7.25 | 8.34 (4B 4-bit) |
| 9B, 3-bit | 3.92 GB | 7.90 | 7.11 | 7.03 | 7.85 (9B 3-bit) |
| 9B, 4-bit | 5.04 GB | 7.16 | 7.01 | 6.99 | 7.16 (9B 4-bit) |

Every best single fit above is itself rotated and scaled. In every setting at every size,
2-bit included, any three of four scaled children beat the best single fit. Unscaled, that
held only at 3 and 4 bits. The margins narrow somewhat as the single models improve too (9B
3-bit: 0.75, against 0.82 unscaled), but stay large at 3 bits. The 9B 2-bit fleet is the
clearest case of nodes jointly running a model none of them could hold: four 2.8 GB nodes
rebuild to 7.25, close to the 7.16 of a scaled 9B 4-bit model that needs 5 GB.

Without scaling ([sizes](sizes/sizes.md)), the conclusions are the same at 3 and 4 bits,
but the single surviving node falls behind the best fit by 0.11–0.29 (4-bit, 2B), 0.75–1.2
(3-bit, 4B) and 1.2–1.8 (3-bit, 9B), and 2-bit fleets beat it only with all four alive.

## What can be concluded

1. **Functional shards are real from 4B up, from 2B up with scaling, and at 4 bits from
   0.8B up.** Each 3- or 4-bit child of a 4B or 9B model is a usable model at 3.6–4.6x less
   memory than bf16, and together they reconstruct the parent nearly losslessly. The two
   properties that replication and erasure coding trade off are both present.
2. **The scheme composes with better quantizers.** Channel scaling improves children,
   rebuilds and single models together. With it and rotation, a staggered 4-bit child at 9B
   is within 0.014 of the same-size single model across rotation seeds, and a 3-bit child
   within 0.05–0.09. The scheme's one weakness, the last node standing, is small where it
   matters.
3. **Bit width is the main design dial, and it sets the trade-off between solo and fleet.**
   At 4 bits, a child alone is nearly as good as the parent, so combining adds little (9B
   scaled: 7.16 → 6.99). At 3 bits, combining recovers most of the gap (7.90 → 7.03). At
   2 bits, children only make sense as a fleet, and with scaling a four-child fleet is
   within 0.3–0.5 of bf16 at 4B and 9B.
4. **Use *N* ≥ 4.** *N* = 2 leaves the fleet a single child after one loss and fails H3 at
   3 bits below 4B; *N* = 4 is within 0.2 of bf16 at 3 bits from 2B up (φ = ½ with
   rotation) and degrades strictly with every loss; *N* = 8 adds headroom mainly for 2-bit
   children.
5. **Degradation is predictable from offsets alone.** Quality depends on how many children
   survive and on the largest gap between their offsets, as the partial-rebuild bound says.
   Any *k* + 1 children beat any *k*, and among equal counts, evenly spaced survivors are
   best.
6. **Phase and rotation are second-order at 4 bits; scaling is not.** φ = ½ and rotation
   change 4-bit perplexity by hundredths at 4B and 9B. Scaling moves the worst 4-bit child
   by 0.1–0.2 and the worst 3-bit child by 1.0–1.4 at the same sizes.
7. **What cannot be concluded yet.** Scaling at *N* ≠ 4 or φ = 0, other group sizes,
   comparison with GPTQ-style children, or speed on real hardware. Measurement noise is
   below 0.005; rotation seeds move single 9B models by up to about 0.03. Task accuracy and a
   speed model are in the [swarm report](swarm-report.md).

## Why it matters

A deployed model is normally a single artifact. Today's protection options force a choice:
**replicas** serve during an outage but cost a full model each and gain nothing from being
many; **erasure-coded shards** are cheap and exact but serve nothing until decoded. These
results show a third option that gets most of both, without any training:

- **Availability during an outage.** Every surviving node can keep answering with a model
  within 4% (4-bit) or 20% (3-bit) of the parent's perplexity at 4B and 9B with scaling and
  rotation.
- **Recovery without a backup.** The parent comes back from the children alone, exactly and
  verifiably by hash, and from a partial set at reduced but predictable precision.
- **Capacity from many small devices.** A set of devices that each hold a few gigabytes can
  jointly run a model better than any one of them could hold: four 2.8 GB nodes rebuild a
  9B model to 7.25, where the best model that fits on one of them scores 8.34.
- **Little cost.** With scaling and rotation, the scheme gives up at most 0.014 (4-bit) or
  0.05–0.09 (3-bit) against the same-size single model at 9B even when only one node is
  left, and adds no memory.

The scaling direction also matters: the scheme gets more useful as models grow (3-bit
children become usable between 2B and 4B, or at 2B with channel scaling), which is where
protecting and distributing a model is most expensive.

## Implications for phase 2: the orchestrated system

*Written before the reframe and kept as written. Phase 2 became a swarm of agents studied in
simulation (see the [swarm pre-registration](../preregistration-swarm.md) and
[swarm report](swarm-report.md)); the container demo below is deferred to follow-up systems
work. Points 3 and 6 were since confirmed: the placement rule at 4B, and stage-wise
additivity (H6c) at 0.8B and 2B.*

Phase 2 puts the scheme under an orchestration layer: a deterministic control plane (health,
leader election, rebuild, revoke; stage 2.3), containers (2.4), a router that chooses which
model answers each request (the LangGraph agent layer, 2.6) and pooled stage workers (2.7).
The phase 1 results translate into concrete inputs and constraints for that layer.

### 1. The router has a quality ladder it can rely on

Every state the orchestrator can be in now has a measured quality. 4-bit children, *N* = 4,
φ = ½ with rotation:

| Serving state | 9B | 9B scaled | 4B | 4B scaled |
|---------------|---:|----------:|---:|----------:|
| Main healthy (bf16 parent) | 6.98 | 6.98 | 8.06 | 8.06 |
| Rebuilt or pooled, all 4 children | 7.00 | 6.99 | 8.07 | 8.07 |
| Rebuilt or pooled, 3 of 4 | 7.04 | 7.01 | 8.12 | 8.10 |
| Solo, best child | 7.28 | 7.15 | 8.43 | 8.33 |
| Solo, worst child | 7.34 | 7.16 | 8.47 | 8.36 |

Two of four was measured only at 0.8B, where it sits between three of four and one child in
every setting (4-bit: 13.46–13.77 against 13.32–13.34 and 14.61–14.85). The router's choice
is therefore mostly **latency versus a known quality gap**, not "working versus broken".
Two consequences:

- *Which* child serves solo barely matters at 4 bits (best and worst within 0.06 unscaled,
  0.04 scaled), so the control plane's deterministic "lowest live index wins" election needs
  no quality term. Unscaled 3-bit children spread by up to 0.6 at 9B, which would argue for
  preferring the child with offset ½; with scaling and rotation the spread is 0.05, and the
  election can stay quality-blind at 3 bits too (scaled but unrotated, the spread is 0.2).
- The pooled mode's advantage over solo is real but modest at 4 bits (about 0.17 perplexity
  at 9B with scaling, 0.3 without). The demo should measure whether that is worth pooled
  mode's per-layer communication, which is the main unknown phase 1 cannot answer.

The pooled-mode losses use the *mean* estimator, not the midpoint measured here; the pool
screen suggests the two differ by under 0.01 at 0.8B, but the full pool run should confirm
this before the ladder is used for routing.

### 2. Rebuild from whatever survives; don't wait

Any *k* + 1 children beat any *k*, so the main container should rebuild as soon as it is
back, from however many children answer, record the missing offsets, and refine when more
return (the "progressive rebuild" in [future work](../future-work.md#systems)). Every
additional child is a strict improvement, so there is no survivor count at which waiting is
better than rebuilding; the only threshold the router needs is the one where pooled serving
stops being worth its communication cost against solo.

### 3. Place children so that a failure domain takes out evenly spaced offsets

Because two opposite children rebuild much better than two adjacent ones, *where* children
live matters as much as how many there are. If four children share two failure domains (two
hosts, racks or power circuits), the roster should put children 0 and 2 in one domain and 1
and 3 in the other, so that each domain holds offsets ½ apart. Losing a domain then leaves
an opposite pair (0.8B 3-bit: 14.90–15.02) instead of an adjacent one (16.47–16.96). In
general each domain should hold an evenly spaced subset of offsets, so the survivors of any
domain loss still are. This is a static rule for the operator-written roster of stage 2.7,
so it keeps the plan's division intact: placement is decided by the operator, never
discovered or changed by the nodes.

### 4. Channel scaling is the demo's biggest cheap gain, but needs an export path

llama.cpp has no 2- or 3-bit block format and no kernel for rotated weights, so the
containerized children are **4-bit, unrotated, *g* = 32** (Q4_1). Rotation is worth
hundredths at 4 bits from 4B up, so losing it costs little. Scaling is worth more: unrotated
4-bit children with φ = ½ improve from 11.02 to 10.73 at 2B and from 7.37 to 7.21 at 9B.

Two pieces are missing before the demo can use it:

- **Shards cannot store the scales yet.** `syncytium-shard` refuses scaled schemes. The
  scales are one fp16 vector per weight, shared by every child, so they belong in the
  manifest or alongside the unquantized tensors, and must be covered by the parent hash.
- **llama.cpp has no per-input-channel divisor.** AWQ exports handle this by folding 1/*s*
  into the operation that produces the layer's input (a norm or the previous linear layer)
  before quantizing. Folding into a norm is shared by every child and keeps the rebuild
  exact; whether every quantized input in Qwen3.5's Gated DeltaNet and attention blocks has
  such a predecessor needs to be checked layer by layer. The PyTorch pooled worker can apply
  the scales directly and does not need folding.

Beyond scaling, the options from before still stand:

- Accept 4-bit as the "cheap insurance" configuration: solo children are within 0.04 of the
  RTN model of the same size unscaled (4B 8.51 against 8.48, 9B 7.37 against 7.33), and
  within 0.03 with scaling and rotation.
- Run the pooled PyTorch workers (stages 2.4 and 2.7) with **3-bit** children, which do not
  depend on llama.cpp and can use rotation and scaling. That shows the regime where the fleet
  clearly beats any single node (9B: 7.11 with three of four against 7.85 for the best
  single fit), and with scaling the solo fallback is no longer poor (worst child 7.90).
- New with scaling: **2-bit pooled-only fleets.** Four scaled 2-bit 9B children of 2.8 GB
  each rebuild or pool to 7.25. No single child is usable (21–26), so the router must never
  send a request to a lone 2-bit child; this is a configuration where solo mode does not
  exist and pooling is the whole point.

Either way, *g* = 32 and the fp16-rounded Q4_1 offsets were not part of the sweep, so the
demo should re-measure perplexity on the exported children rather than reuse phase 1
numbers.

### 5. Model size for the demo: 2B fits the local machine

The 0.8B model is the weakest case in every table (4-bit children 1.09–1.17x, 3-bit unusable
even with scaling). **2B is the natural local demo model:** its 4-bit children are
1.05–1.10x bf16, its scaled 3-bit children pass H2 (1.30x), four 4-bit children rebuild
within 0.03–0.06, a 4-bit *g* = 32 child takes about 1.2 GB (about 4.7 GB for four), and the
bf16 parent (3.8 GB of weights) fits the 10 GB RTX 3080 that serves the main container. 4B
would need about 2.6 GB per child and an 8.4 GB bf16 parent, which leaves the local GPU no
room to spare, so 4B and 9B demos belong on the cluster.

### 6. Predictable degradation enables quality-aware health reporting

Phase 1 now shows that a rebuild's quality depends on which offsets are present in a
predictable way. If the full pool run also confirms H6c (stage-wise additivity of
log-perplexity, R² ≈ 1.0 in the screen), the control plane can report a **predicted
perplexity for the current device state** from a small table of measurements, without
evaluating anything at run time. The router then routes on a number rather than a heuristic
like "enough children are reachable". This keeps the division the plan sets out: the
control plane measures and reports deterministically; the agent layer only decides which
model answers.

### 7. What phase 2 must measure that phase 1 could not

- Time to detect a failure, to fail over, and to rebuild, on real low-bit weights. Stage 2.1
  rebuilds the 0.8B set in about 13 s on CPU with a 3.8 GB peak; 2B, 4B and 9B sets are
  about 2.5x, 6x and 12x larger, so rebuild time should be measured, not assumed.
- Pooled mode's bytes per token and tokens per second against solo, and total compute per
  token across nodes.
- Task-level quality through the agent layer before, during and after a failure, since
  perplexity differences of 0.2–0.3 may or may not show up as task failures.

## Gaps and next steps

| Gap | Effect | Remedy | Command |
|-----|--------|--------|---------|
| ~~Pooled pipeline only screened~~ | Done: H6 holds at 0.8B and 2B | — | `configs/pool-0.8b.toml`, `configs/pool-2b.toml` |
| ~~Two or more missing children measured only at 0.8B~~ | Done: ordering and placement rule hold at 4B | — | `configs/subsets-4b.toml` |
| ~~One rotation seed~~ | Done: the 3-bit "within 0.05" claim fails across seeds | — | `configs/seed1-9b.toml`, `configs/seed2-9b.toml` |
| ~~Demo scheme (*g* = 32) never measured~~ | Done: strict ordering in all four settings ([demo-2b](demo-2b/summary.md)) | — | `configs/demo-2b.toml` |
| Shards and GGUF export do not support channel scales | The demo cannot use the largest gain phase 1 found | Store scales in the shard set and hash them; check folding for every Qwen3.5 layer type | Code (stages 2.1, 2.2) |
| Scaling measured only at *N* = 4, φ = ½ | *N* = 8 with scaling is inferred | Low priority: *N* = 4 is the demo's setting | None |
| ~~Perplexity only~~ | GSM8K and MMLU at 4B and 9B, 3-bit scaled ([swarm report](swarm-report.md)) | — | `configs/swarm-*.toml` |
| No GPTQ-style reference | Single models compared with RTN and AWQ-style scaling only | Phase 3, as a single-model reference point | Not implemented |

Every listed config submits with `bash slurm/submit.sh <config>`; pool configs run through
`syncytium-pool` automatically.

## Data and figures

| Result | Table | Figures |
|--------|-------|---------|
| 0.8B pilot | [summary](pilot-0.8b/summary.md), [baselines](pilot-0.8b/baselines.md) | [recovery](pilot-0.8b/recovery-p0.5-rot-on.png), [Pareto](pilot-0.8b/pareto-p0.5-rot-on.png), [offsets](pilot-0.8b/offsets.png), [baselines](pilot-0.8b/baselines-p0.5-rot-on.png) |
| 0.8B, every survivor count | [summary](subsets-0.8b/summary.md), [baselines](subsets-0.8b/baselines.md) | [recovery](subsets-0.8b/recovery-p0.5-rot-on.png), [baselines](subsets-0.8b/baselines-p0.5-rot-on.png) |
| 2B sweep | [summary](sweep-2b/summary.md), [baselines](sweep-2b/baselines.md) | [recovery](sweep-2b/recovery-p0.5-rot-on.png), [Pareto](sweep-2b/pareto-p0.5-rot-on.png), [offsets](sweep-2b/offsets.png), [baselines](sweep-2b/baselines-p0.5-rot-on.png) |
| 4B sweep | [summary](sweep-4b/summary.md), [baselines](sweep-4b/baselines.md) | [recovery](sweep-4b/recovery-p0.5-rot-on.png), [Pareto](sweep-4b/pareto-p0.5-rot-on.png), [offsets](sweep-4b/offsets.png), [baselines](sweep-4b/baselines-p0.5-rot-on.png) |
| 9B sweep | [summary](sweep-9b/summary.md), [baselines](sweep-9b/baselines.md) | [recovery](sweep-9b/recovery-p0.5-rot-on.png), [Pareto](sweep-9b/pareto-p0.5-rot-on.png), [offsets](sweep-9b/offsets.png), [baselines](sweep-9b/baselines-p0.5-rot-on.png) |
| Channel scaling, 0.8B | [summary](scaling-0.8b/summary.md), [baselines](scaling-0.8b/baselines.md) | [recovery](scaling-0.8b/recovery-p0.5-rot-on-scaled.png), [Pareto](scaling-0.8b/pareto-p0.5-rot-on-scaled.png) |
| Channel scaling, 2B | [summary](scaling-2b/summary.md), [baselines](scaling-2b/baselines.md) | [recovery](scaling-2b/recovery-p0.5-rot-on-scaled.png), [Pareto](scaling-2b/pareto-p0.5-rot-on-scaled.png) |
| Channel scaling, 4B | [summary](scaling-4b/summary.md), [baselines](scaling-4b/baselines.md) | [recovery](scaling-4b/recovery-p0.5-rot-on-scaled.png), [Pareto](scaling-4b/pareto-p0.5-rot-on-scaled.png) |
| Channel scaling, 9B | [summary](scaling-9b/summary.md), [baselines](scaling-9b/baselines.md) | [recovery](scaling-9b/recovery-p0.5-rot-on-scaled.png), [Pareto](scaling-9b/pareto-p0.5-rot-on-scaled.png) |
| Equal memory across sizes | [unscaled](sizes/sizes.md), [scaled](sizes-scaled/sizes.md) | [unscaled](sizes/sizes-p0.5-rot-on.png), [scaled](sizes-scaled/sizes-p0.5-rot-on-scaled.png) |
| 4B, every survivor count | [summary](subsets-4b/summary.md), [baselines](subsets-4b/baselines.md) | [recovery](subsets-4b/recovery-p0.5-rot-on-scaled.png), [baselines](subsets-4b/baselines-p0.5-rot-on-scaled.png) |
| Rotation seeds 1 and 2, 9B | [seed 1](seed1-9b/summary.md), [seed 2](seed2-9b/summary.md) | [seed 1 recovery](seed1-9b/recovery-p0.5-rot-on-scaled.png), [seed 2 recovery](seed2-9b/recovery-p0.5-rot-on-scaled.png) |
| Demo scheme, 2B | [summary](demo-2b/summary.md), [baselines](demo-2b/baselines.md) | [recovery](demo-2b/recovery-p0.5-rot-off-scaled.png) |
| Pooled pipeline | [0.8B](pool-0.8b/summary.md), [2B](pool-2b/summary.md) | — |

The sweep figures exist for all four (φ, rotation) combinations and the scaling figures for
all four (rotation, scaling) combinations at φ = ½; the rotated (and scaled) versions are
linked because they are the best-performing settings. Raw logs are in
`results/<name>/results.jsonl`, with the config, commit and package versions of every run in
`runs.jsonl`.
