# Phase 1 Report: Staggered Children at 0.8B, 4B and 9B

*Draft, 2026-10-17. Covers the 0.8B pilot, the 4B and 9B sweeps, the same-node baselines,
the equal-memory comparison across model sizes and the pooled-pipeline screen. The 2B sweep
has not run.*

## Summary

- **The scheme works as designed, and it gets better as the model grows.** At 4B and 9B,
  every 3- and 4-bit child is a usable model on its own (worst child at most 1.41x bf16
  perplexity), and four children rebuild a parent within 0.03 (4-bit) or 0.13 (3-bit) of
  bf16. At 0.8B, only 4-bit children are usable.
- **Losing a child costs little, and losing children degrades gracefully.** In every setting
  at every size, any *N* − 1 children rebuild a better model than the best single child.
- **Children that are useless alone still carry the parent.** Eight 2-bit children, each
  with a perplexity in the thousands, rebuild the 9B model to within 0.11–0.15 of bf16.
- **A fleet of small nodes beats any single model that fits on one of them.** At the memory
  of one child, any three of four 3- or 4-bit children rebuild a better model than the best
  single model of any measured size that fits in that memory. At 9B with 3-bit children
  (3.9 GB per node), three of four give 7.25 and four give 7.08; the best model that fits in
  3.9 GB is 4B at 6 bits, at 8.07.
- **The price is paid by the last node standing.** With one node left, a staggered child is
  slightly worse than the round-to-nearest (RTN) model of the same size: by at most 0.05 at
  4 bits (4B and 9B), but by 0.3–1.8 at 3 bits once the alternative of a smaller model at
  more bits is counted.
- **For phase 2:** the 4-bit children the container demo can export are the setting where
  the solo fallback is nearly free (within 0.04–0.05 of RTN at 4B and 9B) but where pooling
  and rebuilding gain least. The setting where combining matters most, 3 bits, is not
  servable by llama.cpp. This shapes the router's policy and which modes the demo can show
  convincingly; see [implications for phase 2](#implications-for-phase-2-the-orchestrated-system).

## What was run

| Item | Setting |
|------|---------|
| Models | Qwen3.5 Base at 0.8B (0.75B parameters), 4B (4.21B) and 9B (8.95B) |
| Metric | WikiText-2 test perplexity, non-overlapping 2048-token windows over the whole split |
| Grid | *b* ∈ {2, 3, 4}, *N* ∈ {2, 4, 8}, *g* = 64, phase φ ∈ {0, ½}, rotation off/on (seed 0) |
| Conditions per setting | bf16; every child alone; full rebuild; every rebuild with one child missing; RTN at 2–8 bits |
| Hardware | RTX 3080 (0.8B), H100 PCIe (4B, 9B); about 2.6–3.1 GPU-hours per sweep |
| Totals | 387 logged conditions per sweep, 1,161 across the three |

Phase 1 swaps dequantized weights into a bf16 model, so it measures quality and memory, not
speed. Memory figures count codes, fp16 scales and every unquantized tensor. Rebuilds with
children missing use the interval-midpoint estimator ([method](../method.md#partial-rebuild)).

Reference perplexities (bf16): **0.8B 13.125, 4B 8.055, 9B 6.982**.

## Results by hypothesis

Thresholds were fixed after the 0.8B pilot and before any 4B or 9B result was examined
([hypothesis](../hypothesis.md#what-results-would-mean)): H2 needs the worst child within
1.5x of bf16; H3 needs the full rebuild within 0.5 of bf16; H4 is checked for *N* ≥ 3.

### H1: exact superposition — holds

The rebuild identity is covered by unit tests (with and without rotation, both phases), and
the shard writer of stage 2.1 rebuilds a 0.8B set from packed files with a matching SHA-256
of the parent's codes. This is an implementation check; the identity itself is Hermite's.

### H2: functional children — holds at 4 bits everywhere, at 3 bits from 4B

Worst child as a multiple of bf16 perplexity, over the four (φ, rotation) settings:

| Bits | Memory vs bf16 | 0.8B | 4B | 9B |
|-----:|---------------:|-----:|---:|---:|
| 4 | 3.6x smaller | 1.12–1.17 ✅ | 1.05–1.06 ✅ | 1.05–1.06 ✅ |
| 3 | 4.6x smaller | 1.97–2.70 ❌ | 1.31–1.40 ✅ | 1.32–1.41 ✅ |
| 2 | 6.4x smaller | > 2,000 ❌ | > 400 ❌ | > 400 ❌ |

The jump from 0.8B to 4B is the main scaling result: 3-bit children cross the threshold.
The ratio does not improve further from 4B to 9B, so "bigger is better" should not be
extrapolated without more sizes.

### H2a: rotation helps low-bit children — holds for children, not rebuilds

The shared Hadamard rotation lowers the worst 3-bit child at every size (0.8B 32.0 → 27.2,
4B 11.07 → 10.82, 9B 9.61 → 9.27, all at φ = ½) and the 2-bit children by 3–10x (still
unusable). At 4 bits it moves children by 0.01–0.05 at 4B and 9B. Full rebuilds move by
similar small amounts in either direction. At 4B and 9B, 3-bit children pass H2 *without*
rotation, so rotation is a refinement there rather than an enabler. φ = ½ with rotation gives
the best worst-child in most settings.

### H3: near-lossless rebuild — holds whenever the rebuild has about 5+ bits

Full-rebuild gap to bf16 (perplexity), φ = ½ with rotation:

| Bits | *N* | Parent bits | 0.8B | 4B | 9B |
|-----:|----:|------------:|-----:|---:|---:|
| 4 | 2 | 5 | 0.35 ✅ | 0.09 ✅ | 0.08 ✅ |
| 4 | 4 | 6 | 0.07 ✅ | 0.02 ✅ | 0.02 ✅ |
| 4 | 8 | 7 | 0.03 ✅ | 0.01 ✅ | 0.00 ✅ |
| 3 | 2 | 4 | 1.74 ❌ | 0.43 ✅ | 0.38 ✅ |
| 3 | 4 | 5 | 0.39 ✅ | 0.10 ✅ | 0.10 ✅ |
| 3 | 8 | 6 | 0.09 ✅ | 0.03 ✅ | 0.02 ✅ |
| 2 | 4 | 4 | 2.77 ❌ | 0.65 ❌ | 0.58 ❌ |
| 2 | 8 | 5 | 0.54 ❌ | 0.14 ✅ | 0.11 ✅ |

Across all settings: every 4-bit setting passes; 3-bit passes everywhere at 9B, for *N* ≥ 4
at 4B, and for *N* ≥ 4 at 0.8B except one setting; 2-bit passes only with eight children at
4B and 9B. The gap shrinks by roughly 4x or more each time *N* doubles until it reaches a
floor of a few thousandths, consistent with the rebuild's error falling as 1/*N*
([method](../method.md#rebuild)) and the perplexity gap tracking squared error.

### H4: graceful degradation — holds in every tested setting

For every *N* ≥ 4 setting at every size, the worst rebuild from *N* − 1 children beats the
best single child. Losing one of four children costs, at 9B: 0.04 (4-bit), 0.17 (3-bit) and
1.2 (2-bit) perplexity; losing one of eight costs at most 0.01, 0.02 and 0.12. **Caveat:**
only one missing child was measured (`max_missing = 1`). Monotonicity beyond one loss, and
two-of-four rebuilds in particular, are untested; `configs/subsets-0.8b.toml` exists for this
but has not run.

### H5: survival in a running system — not yet tested

This is phase 2. The pieces it depends on that phase 1 can vouch for are the exact rebuild
(H1) and the quality of every fallback state (H2–H4).

### H6: pooled pipeline — pipeline check only

The `pool-screen-0.8b` run (2 stages, 4 devices each, 8 windows) exercised the code end to
end. On its slice, losing any one device stays within 0.10 of bf16, and the log-perplexity of
multi-device losses is predicted by summing per-stage effects with R² ≈ 1.0. With 8 windows
and 4 random patterns this is not evidence for H6; the full `pool-0.8b` run is pending.

## Comparison with simpler schemes

### Same nodes, same memory

Worst-case perplexity with *s* of 4 nodes alive, 9B, φ = ½ with rotation
([baselines table](sweep-9b/baselines.md)):

| Scheme | Extra memory / node | 3-bit, *s* = 1 | 3-bit, *s* = 3 | 3-bit, *s* = 4 | 4-bit, *s* = 1 | 4-bit, *s* = 3 | 4-bit, *s* = 4 |
|--------|--------------------:|------:|------:|------:|------:|------:|------:|
| Staggered children | 0 | 9.27 | **7.25** | **7.08** | 7.34 | **7.04** | **7.00** |
| Same RTN model on every node | 0 | **8.92** | 8.92 | 8.92 | **7.30** | 7.30 | 7.30 |
| Base child + coded remainder, *k* = 3 | +0.67 bpw | 8.92 | 7.08 | 7.08 | 7.30 | 7.00 | 7.00 |

- **Replication** wins only with one node left. With three of four, it needs about 2 more
  bits per node to match staggered children.
- **The coded remainder** rebuilds the exact parent from any *k* nodes for log₂(*N*)/*k*
  extra bits per node, and serves the base child below *k*. It beats staggered at *s* ≥ *k*
  if that memory is available; with all four alive, staggered matches it at no extra memory.
- **Staggered children are the only scheme whose quality rises with every surviving node at
  zero extra memory**, and the only one whose shards can cooperate without first being
  decoded (pooling). 4B and 0.8B show the same pattern.

### Equal memory, any model size

At the weight memory of one child, compared with the best single model of any measured size
(bf16 or RTN at 2–8 bits) that fits ([sizes table](sizes/sizes.md),
[figure](sizes/sizes-p0.5-rot-on.png)); *N* = 4, φ = ½, rotation:

| Setting | Memory / node | One node alive | Any 3 of 4 | All 4 | Best single fit |
|---------|--------------:|---------------:|-----------:|------:|----------------:|
| 4B, 3-bit | 1.84 GB | 10.82 | 8.34 | 8.16 | 10.33 (4B 3-bit) |
| 4B, 4-bit | 2.37 GB | 8.47 | 8.12 | 8.07 | 8.47 (4B 4-bit) |
| 9B, 2-bit | 2.80 GB | 6,220 | 8.73 | 7.56 | 8.47 (4B 4-bit) |
| 9B, 3-bit | 3.92 GB | 9.27 | 7.25 | 7.08 | 8.07 (4B 6-bit) |
| 9B, 4-bit | 5.04 GB | 7.34 | 7.04 | 7.00 | 7.30 (9B 4-bit) |

In every 3- and 4-bit setting at every size, any three of four children beat the best single
fit. The single surviving node falls behind it by 0.5–0.9 (3-bit, 4B) and 1.2–1.8 (3-bit,
9B), but by at most about 0.05 at 4 bits. With no 2B model, there is no candidate between
0.8B and 4B, which flatters the 0.8B rows; the 9B 2-bit fleet is the clearest case of nodes
jointly running a model none of them could serve alone.

## What can be concluded

1. **Functional shards are real from 4B up.** Each 3- or 4-bit child is a usable model at
   3.6–4.6x less memory than bf16, and together they reconstruct the parent nearly losslessly.
   The two properties that replication and erasure coding trade off are both present.
2. **Bit width is the main design dial, and it sets the trade-off between solo and fleet.**
   At 4 bits, a child alone is nearly as good as the parent, so combining adds little (9B:
   7.34 → 7.00). At 3 bits, a child alone is clearly worse, and combining recovers most of
   the gap (9.27 → 7.08). At 2 bits, children only make sense as a fleet.
3. **Use *N* ≥ 4.** *N* = 2 leaves the fleet a single child after one loss and fails H3 at
   3 bits for small models; *N* = 4 is within 0.1 of bf16 at 3 bits (4B, 9B) and tolerates
   any single loss at small cost; *N* = 8 adds headroom mainly for 2-bit children.
4. **Phase and rotation are second-order at 4 bits.** φ = ½ and rotation help the worst
   child, mostly at 2–3 bits. At 4 bits they change perplexity by hundredths.
5. **What cannot be concluded yet.** Anything about two or more simultaneous losses, other
   group sizes, downstream task accuracy, variance across rotation seeds (one seed per
   setting), behavior between 0.8B and 4B, comparison with GPTQ/AWQ children, or speed.
   Differences below about 0.05 perplexity at 4 bits come from single evaluations and should
   not be ranked.

## Why it matters

A deployed model is normally a single artifact. Today's protection options force a choice:
**replicas** serve during an outage but cost a full model each and gain nothing from being
many; **erasure-coded shards** are cheap and exact but serve nothing until decoded. These
results show a third option that gets most of both, without any training:

- **Availability during an outage.** Every surviving node can keep answering with a model
  within 5% (4-bit) or 41% (3-bit) of the parent's perplexity at 4B and 9B.
- **Recovery without a backup.** The parent comes back from the children alone, exactly and
  verifiably by hash, and from a partial set at reduced but predictable precision.
- **Capacity from many small devices.** A set of devices that each hold a few gigabytes can
  jointly run a model better than any one of them could hold, for example a 9B-class model
  on 3.9 GB nodes that beats every single model fitting in 3.9 GB by about 0.8 perplexity.
- **The cost is concentrated in one case.** The scheme gives up a little quality only when
  exactly one node is left, and almost nothing at 4 bits.

The scaling direction also matters: the scheme gets more useful as models grow (3-bit
children become usable between 0.8B and 4B), which is where protecting and distributing a
model is most expensive.

## Implications for phase 2: the orchestrated system

Phase 2 puts the scheme under an orchestration layer: a deterministic control plane (health,
leader election, rebuild, revoke; stage 2.3), containers (2.4), a router that chooses which
model answers each request (the LangGraph agent layer, 2.6) and pooled stage workers (2.7).
The phase 1 results translate into concrete inputs and constraints for that layer.

### 1. The router has a quality ladder it can rely on

Every state the orchestrator can be in now has a measured quality. At 9B, 4-bit children,
*N* = 4 (4B in brackets):

| Serving state | Perplexity | Gap to bf16 |
|---------------|-----------:|------------:|
| Main healthy (bf16 parent) | 6.98 (8.06) | — |
| Rebuilt or pooled, all 4 children | 7.00 (8.07) | 0.02 (0.02) |
| Rebuilt or pooled, 3 of 4 | 7.04 (8.12) | 0.06 (0.06) |
| Solo, worst child | 7.34 (8.47) | 0.35 (0.41) |
| Solo, best child | 7.28 (8.43) | 0.30 (0.37) |

The router's choice is therefore mostly **latency versus a known quality gap**, not
"working versus broken". Two consequences:

- *Which* child serves solo barely matters at 4 bits (best and worst within 0.06), so the
  control plane's deterministic "lowest live index wins" election needs no quality term. At
  3 bits the spread is up to 0.6 at 9B, and the child with offset ½ (the RTN child) should
  be preferred.
- The pooled mode's advantage over solo is real but modest at 4 bits (about 0.3 perplexity
  at 9B). The demo should measure whether that is worth pooled mode's per-layer
  communication, which is the main unknown phase 1 cannot answer.

The pooled-mode losses use the *mean* estimator, not the midpoint measured here; the pool
screen suggests the two differ by under 0.01 at 0.8B, but the full pool run should confirm
this before the ladder is used for routing.

### 2. Rebuild from whatever survives; don't wait

H4 holds everywhere tested: any *N* − 1 rebuild beats every single child. The lifecycle's
partial-rebuild step (method, step 5) should therefore start as soon as the main container
is back with *N* − 1 children, record the missing offset, and refine when the last child
returns (the "progressive rebuild" in [future work](../future-work.md#systems)). The data
does not yet say where to draw the line below *N* − 1; running `subsets-0.8b` (two-of-four
rebuilds) before the router's thresholds are fixed would close that gap.

### 3. The demo runs in the regime where pooling gains least

llama.cpp has no 2- or 3-bit block format and no kernel for rotated weights, so the
containerized children are **4-bit, unrotated, *g* = 32** (Q4_1). Phase 1 says that costs
little for solo serving (rotation is worth hundredths at 4 bits), but it also means the demo
shows the scheme where combining adds least. Options, in order of effort:

- Accept it, and state that 4-bit is the "cheap insurance" configuration: solo children are
  within 0.04–0.05 of a dedicated RTN deployment of the same size, and the rebuild is a
  bonus.
- Run the pooled PyTorch workers (stages 2.4 and 2.7) with **3-bit** children, which do not
  depend on llama.cpp. That shows the regime where the fleet clearly beats any single node
  (9B: 7.25 with three of four against 8.07 for the best single fit), at the cost of a slow
  solo fallback.

Either way, *g* = 32 and the fp16-rounded Q4_1 offsets were not part of the sweep, so the
demo should re-measure perplexity on the exported children rather than reuse phase 1
numbers.

### 4. Model size for the demo

The 0.8B model is the weakest case in every table (4-bit children 1.12–1.17x, 3-bit
unusable, and the "best single fit" comparison is flattered by the missing 2B). A 0.8B demo
proves the mechanics of kill, wipe and rebuild; quality claims should cite 4B or 9B. A 4B
demo needs about 2.6 GB per 4-bit *g* = 32 child (four children, about 10.5 GB of CPU
memory in total) and a main container that can serve the 8.4 GB bf16 parent, which the
10 GB local GPU cannot do comfortably; the cluster or a quantized main is needed for that.

### 5. Predictable degradation enables quality-aware health reporting

If the full pool run confirms H6c (stage-wise additivity of log-perplexity, R² ≈ 1.0 in the
screen), the control plane can report a **predicted perplexity for the current device
state** from a small table of per-stage measurements, without evaluating anything at run
time. The router then routes on a number rather than a heuristic like "enough children are
reachable". This keeps the division the plan sets out: the control plane measures and
reports deterministically; the agent layer only decides which model answers.

### 6. What phase 2 must measure that phase 1 could not

- Time to detect a failure, to fail over, and to rebuild, on real low-bit weights. Stage 2.1
  rebuilds the 0.8B set in about 13 s on CPU with a 3.8 GB peak; 4B and 9B sets are roughly
  5x and 12x larger, so rebuild time should be measured, not assumed.
- Pooled mode's bytes per token and tokens per second against solo, and total compute per
  token across nodes.
- Task-level quality through the agent layer before, during and after a failure, since
  perplexity differences of 0.3 may or may not show up as task failures.

## Gaps and next steps

| Gap | Effect | Remedy |
|-----|--------|--------|
| 2B sweep failed (CUDA error 101, before loading) | No point between 0.8B and 4B; flatters 0.8B equal-memory rows | Resubmit with a different node or driver |
| Only one missing child measured | H4 checked only at *m* = 1; router thresholds below *N* − 1 unknown | Run `subsets-0.8b`, then a 4B subset run |
| Pooled pipeline only screened | H6 untested | Run `pool-0.8b`, then 2B or 4B |
| One rotation seed, one evaluation | Cannot rank settings within ~0.05 | Two more seeds for the 3-bit settings |
| Perplexity only | No downstream accuracy | Zero-shot suite on 4B and 9B, 3- and 4-bit, *N* = 4 |
| *g* = 64 only; demo uses *g* = 32 | Demo quality not directly measured | Evaluate exported Q4_1 children |
| No GPTQ/AWQ reference | Children compared only with RTN | Phase 3, as planned |

## Data and figures

| Result | Table | Figures |
|--------|-------|---------|
| 0.8B pilot | [summary](pilot-0.8b/summary.md), [baselines](pilot-0.8b/baselines.md) | [recovery](pilot-0.8b/recovery-p0.5-rot-on.png), [Pareto](pilot-0.8b/pareto-p0.5-rot-on.png), [offsets](pilot-0.8b/offsets.png), [baselines](pilot-0.8b/baselines-p0.5-rot-on.png) |
| 4B sweep | [summary](sweep-4b/summary.md), [baselines](sweep-4b/baselines.md) | [recovery](sweep-4b/recovery-p0.5-rot-on.png), [Pareto](sweep-4b/pareto-p0.5-rot-on.png), [offsets](sweep-4b/offsets.png), [baselines](sweep-4b/baselines-p0.5-rot-on.png) |
| 9B sweep | [summary](sweep-9b/summary.md), [baselines](sweep-9b/baselines.md) | [recovery](sweep-9b/recovery-p0.5-rot-on.png), [Pareto](sweep-9b/pareto-p0.5-rot-on.png), [offsets](sweep-9b/offsets.png), [baselines](sweep-9b/baselines-p0.5-rot-on.png) |
| Equal memory across sizes | [sizes](sizes/sizes.md) | [φ = ½, rotation](sizes/sizes-p0.5-rot-on.png) |

Each figure exists for all four (φ, rotation) combinations; the φ = ½ with rotation versions
are linked because that is the best-performing setting in most comparisons. Raw logs are in
`results/<name>/results.jsonl`, with the config, commit and package versions of every run in
`runs.jsonl`.
