# Hypothesis and Goals

> **Reframed after phase 1.** The project now studies a swarm of agents that answer alone, fuse
> on demand and regrow lost members. Those hypotheses are in the
> [swarm pre-registration](preregistration-swarm.md) (H7–H17) and the
> [disagreement pre-registration](preregistration-disagreement.md) (H8b, H10b). H1–H4 and H6
> stand as fixed below; H5 is superseded by H13–H15. The outcome of every hypothesis is listed
> in the [swarm report](results/swarm-report.md#status-of-every-hypothesis).

## Motivation

A deployed LLM is usually a single artifact on a single host. If that host is lost, the model
is gone until someone restores it from a full backup. The two standard ways to protect it each
give up something:

- **Replication** keeps full copies. Each copy costs the full model's memory.
- **Erasure coding** (for example Reed-Solomon) rebuilds the model exactly from any *k* of *N*
  shards at a fraction of the cost, but the shards are inert bytes. Until the rebuild happens,
  nothing can serve requests.

This project asks whether shards can be **functional**: each one a usable language model that
fits on a small device, while all of them together rebuild the high-precision parent.

## Terminology

| Term        | Meaning                                                                    |
|-------------|----------------------------------------------------------------------------|
| Parent      | The model being protected, represented at high precision.                  |
| Child       | A low-bit quantized copy of the parent with its own rounding offset.       |
| *N*         | Number of children.                                                        |
| *b*         | Bits per weight in a child.                                                |
| Δ           | Quantization step size, shared by all children within a weight group.     |
| Rebuild     | Combining children into the parent by adding their integer codes.          |
| Main        | The node (container) that serves the parent.                               |

## Hypotheses

**H1. Exact superposition.** When the *N* children use offsets *k*Δ/*N* for *k* = 0 … *N*−1,
the sum of their integer codes equals the parent's code on a grid of step Δ/*N*. The rebuild is
exact integer arithmetic, verifiable by hash against a reference quantization.
*This follows from Hermite's identity (see [method](method.md)); the experiment checks the
implementation, not the math.*

**H2. Functional children.** A child with *b* ≤ 4 uses at least 3x less weight memory than the
bf16 parent and remains a usable language model on its own.

**H2a. Rotation helps low-bit children.** Applying one shared orthogonal rotation before
quantization improves 2- and 3-bit children without breaking H1.

**H3. Near-lossless rebuild.** The parent rebuilt from all *N* children has perplexity within
**0.5** of the bf16 model.

**H4. Graceful degradation.** Rebuilding from *N* − *m* children loses quality monotonically
in *m*, and any *N* − 1 children rebuild a model better than any single child.

**H5. Survival in a running system.** In the container demo, after the main container is
destroyed and its filesystem wiped:
- the children keep serving requests throughout the outage, and
- the main container, once restarted, is rebuilt from the children alone, with a hash match,
  even though no surviving node holds parent-precision weights.

**H6. Pooled pipeline.** Split the parent into stages, each held by *N* devices that store one
child of that stage (see [method](method.md#pooled-pipeline)). Then:
- **H6a. Exact pooling.** Averaging the outputs of a stage's *N* devices reproduces the
  rebuilt parent exactly. *This follows from the parent being the mean of its children; the
  tests check the implementation.*
- **H6b. One failure is nearly invisible.** Losing any one device, with the stage averaging
  its survivors, keeps perplexity within **0.5** of bf16 (H3's bar).
- **H6c. Predictable degradation.** The log-perplexity increase of a random loss pattern is
  predicted by adding the increases of each affected stage measured on its own, with
  **R² ≥ 0.9** across patterns.

*Proposed on 2026-10-17 and fixed on 2026-10-18, before any pooled sweep beyond the pipeline
check (`pool-screen-0.8b`) was examined. H6b and H6c are evaluated under the mean estimator,
the only one the devices can compute without exchanging codes, and separately on `pool-0.8b`
(4-bit, rotated) and `pool-2b` (3-bit, rotated and scaled); each run passes or fails on its
own. Known when fixing: in the main sweeps, losing one child from every stage at once costs
0.22 and 0.38 perplexity at these two settings (midpoint estimator), so H6b is expected to
pass; H6c is the informative test.* Which stage is most sensitive, and how much the mean
estimator loses against the interval midpoint, are exploratory questions with no threshold.

## Goals

- An open-source reference implementation that is small, tested and easy to read.
- A reproducible perplexity sweep across model sizes, bit widths and child counts.
- A container demo that shows deletion, outage and rebuild end to end.
- A paper reporting the results, including negative ones.

## Non-goals (for this study)

- Beating dedicated quantizers such as GPTQ or AWQ on single-model quality. They appear as
  reference points only.
- Any training or fine-tuning. Children come from post-training quantization alone.
- Bit-exact recovery of the original bf16 weights. The rebuilt parent is the fine-grid model;
  a bit-exact extension is listed in [future work](future-work.md).
- Mixture-of-experts models.

## Evaluation protocol

**Models.** Qwen3.5 Base at 0.8B, 2B, 4B and 9B; the 0.8B pilot ran locally and the rest on a
GPU cluster. Base checkpoints are used because perplexity is the primary metric.

**Configuration grid.**

| Factor            | Levels                     |
|-------------------|----------------------------|
| Bits per child    | 2, 3, 4                    |
| Number of children| 2, 4, 8                    |
| Group size        | 64 (32 and 128 as ablation)|
| Shared rotation   | off, on                    |
| Channel scaling   | off, on (*N* = 4, φ = ½)   |

**Measurements.**

- Perplexity on the WikiText-2 test split (fixed context length and stride, recorded in each
  run's config).
- Weight memory in bytes, including scales and every unquantized tensor.
- Rebuild exactness (hash match) and rebuild wall-clock time, split into verifying the shard
  files and streaming, combining and dequantizing them (`syncytium-rebuild`, logged to
  `rebuilds.jsonl`).
- Zero-shot accuracy on a small task suite for the most promising configurations.

Phase 1 swaps dequantized bf16 weights into one model, so every condition runs at the same
speed; its per-condition `seconds` measure the harness, not the scheme. Speed is measured
only in the container demo, on real low-bit weights and a real network.

**System measurements (container demo).** Every request and event is logged as one JSON line
with a timestamp.

- *Per request:* serving mode (main, solo, pooled, partial rebuild), the children that took
  part, prompt and generated tokens, time to first token, decode tokens per second, end-to-end
  latency, and bytes sent and received per generated token on each node.
- *Per node, sampled during generation:* resident memory and CPU time, against the
  container's memory limit.
- *Per event:* the time a container was killed, how long until the endpoint noticed, how
  many requests failed or were rerouted meanwhile, and for each rebuild the bytes moved, the
  transfer, verification and load times, and the hash result.
- *Quality per mode:* perplexity on a fixed held-out slice, and for pooled mode the largest
  logit difference from the rebuilt parent on the same tokens.
- *Totals:* compute per generated token summed over all nodes, so pooled mode's duplicated
  work is reported next to its latency.

**Conditions compared for each configuration.**

1. bf16 parent (upper bound)
2. Each child alone
3. Rebuild from all *N* children
4. Every rebuild with one child missing, and a sample of rebuilds with more missing
5. Standard round-to-nearest quantization at the child's and the rebuild's bit width
6. GPTQ or AWQ at the child's bit width (reference point)
7. The same nodes holding identical round-to-nearest models, or a base child plus an
   erasure-coded remainder (see [method](method.md#baselines-on-the-same-nodes)), by surviving
   nodes. Added on 2026-10-17 after the 0.8B pilot was examined, as an exploratory comparison
   with no threshold.
8. Children and round-to-nearest models with activation-aware channel scaling (see
   [method](method.md#optional-channel-scaling)), the strongest quantizer that keeps the
   rebuild exact. Added on 2026-10-17 after the 4B and 9B sweeps were examined, as an
   exploratory comparison with no threshold.

## What results would mean

| Outcome                                                  | Interpretation                                   |
|----------------------------------------------------------|--------------------------------------------------|
| H1 holds, H2 and H3 hold at 4 bits only                  | A robustness tool: functional shards at moderate compression. |
| H2 and H3 also hold at 3 bits with rotation              | Strong result: large memory savings and near-lossless rebuild. |
| H2 fails at 4 bits for models of 2B and up               | Children are not useful alone; the scheme reduces to coded storage. |
| H4 fails (one missing child is worse than one child)     | Decoding of partial sets needs rethinking.       |

**Thresholds, fixed on 2026-10-17 after the 0.8B pilot and before any 2B, 4B or 9B result was
examined.** A setting passes H2 when its *worst* child has a WikiText-2 perplexity at most
**1.5x** that of the bf16 parent. The worst child is used because an outage does not choose
which children survive; a ratio is used so the bar means the same at every model size. In the
pilot the worst child of each setting scored 1.12–1.17x at 4 bits, 1.97–2.70x at 3 bits (the
lower end with rotation) and over 2000x at 2 bits. The threshold therefore separates 4 from
3 bits at 0.8B and leaves 3-bit children with rotation as an open question for the larger
models. H4 is evaluated only for *N* ≥ 3, since with *N* = 2 any *N* − 1 children are a
single child. Any later change is recorded here with its reason.

**Criteria for the follow-up runs, fixed on 2026-10-18 before their results were examined.**
These test claims made in the [phase 1 report](results/phase-1-report.md) from a single
rotation seed or a single model size.

- *Seed robustness* (`seed1-9b`, `seed2-9b`, with seed 0 from `scaling-9b`). The claim that a
  scaled, rotated child is within **0.05** perplexity of the scaled, rotated round-to-nearest
  model of the same size at 9B holds if, for each of seeds 0, 1 and 2, the worst child at 3
  and at 4 bits (*N* = 4, φ = ½) is within 0.05 of the round-to-nearest model with the same
  seed. The spread across the three seeds is reported as the uncertainty of every setting
  comparison at 9B.
- *Survivor ordering at 4B* (`subsets-4b`). H4's strict ordering, that the worst rebuild from
  any *k* + 1 of four children beats the best from any *k*, and the ordering of two-child
  rebuilds, that every pair with offsets ½ apart beats every pair with adjacent offsets, each
  hold if they hold in every setting of the run.
- `demo-2b` measures the demo's scheme and has no threshold.

## Origin of the hypothesis

The project began as a proposal to split a dense model into a low-rank shared backbone plus
interleaved SVD slices. That version fails on memory arithmetic and on information theory
(a lost slice cannot be recovered from the others). The reasoning, and the other alternatives
considered, are in the [method](method.md#alternatives-considered).

## Scope

The study concerns fault tolerance on hardware the operator controls. Nodes are enrolled
explicitly, shards never move to unenrolled machines, nodes never start new hosts, and an
operator revoke command stops every node and deletes its shards.
