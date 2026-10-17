# Hypothesis and Goals

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

**Models.** Qwen3.5 Base at 0.8B, 2B and 4B locally; 9B on a GPU cluster. Base checkpoints are
used because perplexity is the primary metric.

**Configuration grid.**

| Factor            | Levels                     |
|-------------------|----------------------------|
| Bits per child    | 2, 3, 4                    |
| Number of children| 2, 4, 8                    |
| Group size        | 64 (32 and 128 as ablation)|
| Shared rotation   | off, on                    |

**Measurements.**

- Perplexity on the WikiText-2 test split (fixed context length and stride, recorded in each
  run's config).
- Weight memory in bytes, including scales and every unquantized tensor.
- Rebuild exactness (hash match) and rebuild wall-clock time.
- Zero-shot accuracy on a small task suite for the most promising configurations.

**Conditions compared for each configuration.**

1. bf16 parent (upper bound)
2. Each child alone
3. Rebuild from all *N* children
4. Every rebuild with one child missing, and a sample of rebuilds with more missing
5. Standard round-to-nearest quantization at the child's and the rebuild's bit width
6. GPTQ or AWQ at the child's bit width (reference point)

## What results would mean

| Outcome                                                  | Interpretation                                   |
|----------------------------------------------------------|--------------------------------------------------|
| H1 holds, H2 and H3 hold at 4 bits only                  | A robustness tool: functional shards at moderate compression. |
| H2 and H3 also hold at 3 bits with rotation              | Strong result: large memory savings and near-lossless rebuild. |
| H2 fails at 4 bits for models of 2B and up               | Children are not useful alone; the scheme reduces to coded storage. |
| H4 fails (one missing child is worse than one child)     | Decoding of partial sets needs rethinking.       |

Numeric thresholds for "usable" in H2 will be fixed after the 0.8B pilot and before the full
sweep. Any later change is recorded here with its reason.

## Origin of the hypothesis

The project began as a proposal to split a dense model into a low-rank shared backbone plus
interleaved SVD slices. That version fails on memory arithmetic and on information theory
(a lost slice cannot be recovered from the others). The reasoning, and the other alternatives
considered, are in the [method](method.md#alternatives-considered).

## Scope

The study concerns fault tolerance on hardware the operator controls. Nodes are enrolled
explicitly, shards never move to unenrolled machines, nodes never start new hosts, and an
operator revoke command stops every node and deletes its shards.
