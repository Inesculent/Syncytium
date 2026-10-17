# Method

## Staggered quantization

Weights are quantized in groups of *g* consecutive values along the input dimension. For one
group with minimum *m*, maximum *M* and child bit width *b*:

- Step size: Δ = (*M* − *m*) / (2^*b* − 1), the standard asymmetric step
- Normalized weight: *x* = (*w* − *m*) / Δ, so *x* lies in [0, 2^*b* − 1]
- Child *k*'s code, for *k* = 0 … *N*−1: *q*ₖ = ⌊*x* + *k*/*N*⌋
- Child *k*'s weight: *ŵ*ₖ = *m* + Δ (*q*ₖ + ½ − *k*/*N*)

Every child uses the same *m* and Δ; only the offset *k*/*N* differs. Each child is an ordinary
asymmetric *b*-bit quantization with worst-case error Δ/2. For even *N*, the child with
*k* = *N*/2 is exactly standard round-to-nearest.

Because *x* + *k*/*N* < 2^*b*, every code fits in [0, 2^*b* − 1]: the shifted grids never
clip, and no quantization level has to be reserved to keep the rebuild exact.

## Rebuild

Hermite's identity states that for any real *x* and positive integer *N*:

  Σₖ ⌊*x* + *k*/*N*⌋ = ⌊*N x*⌋  (sum over *k* = 0 … *N*−1)

So the sum of the children's codes, *Q* = Σₖ *q*ₖ, equals ⌊*N x*⌋, which is the code of the
same weight quantized with step Δ/*N*. The rebuilt weight is

  *ŵ* = *m* + (Δ/*N*)(*Q* + ½)

with worst-case error Δ/(2*N*). The arithmetic mean of the children's weights gives the same
value, but the rebuild is done on integer codes so it is exact and can be checked by hash. The
parent needs about *b* + log₂ *N* bits per weight.

Staggering matters. Averaging children with random offsets (dithering) only reduces the error
as 1/√*N*; staggered offsets reduce it as 1/*N*.

## Partial rebuild

With only a subset of children available, each child still bounds the true value: child *k*
places *x* in the interval [*q*ₖ − *k*/*N*, *q*ₖ + 1 − *k*/*N*). The estimate is the midpoint of
the intersection of these intervals. Its worst-case error is half the largest gap between the
available offsets, in units of Δ. With 4 children and one missing, that is Δ/4: half the error
of a single child.

## What each child stores

- Integer codes for every quantized weight, packed to *b* bits.
- The shared per-group minimum *m* and step Δ (identical in every child).
- Every tensor that is not quantized (norms, Gated DeltaNet convolution and decay parameters),
  in full precision.
- A manifest: model id, *N*, *b*, *g*, the child's offset index *k*, rotation seed if any, and
  hashes of the reference rebuild.

Embeddings are quantized like any other weight. Qwen3.5 uses a 248K-token vocabulary with tied
embeddings, so the embedding table is roughly a quarter of the 2B model and cannot be left in
bf16 if the memory target is to be met.

## Optional shared rotation

Before quantization, each weight matrix can be multiplied by an orthogonal matrix *H* (a
randomized Hadamard transform) along its input dimension: *W*′ = *W H*. Children quantize *W*′,
and weights are recovered as *Ŵ*′*H*ᵀ. *H* is fixed by a seed shared across all children, and
the transform is linear, so the rebuild identity still holds. Rotation spreads outliers across
a group, which usually improves low-bit rounding. In phase 1 it is simulated by dequantizing in
PyTorch; running rotated children efficiently needs kernel support that stock llama.cpp lacks.

## Memory accounting

Bits per weight for a child are *b* + 32/*g* when *m* and Δ are stored as fp16 per group:

| *b* | *g* | Bits per weight | Reduction vs bf16 |
|----:|----:|----------------:|------------------:|
| 4   | 64  | 4.5             | 3.6x              |
| 3   | 64  | 3.5             | 4.6x              |
| 2   | 64  | 2.5             | 6.4x              |

Reported numbers always include unquantized tensors, measured from the files on disk.

## System design (container demo)

**Roles.** One *main* container serves the parent on the GPU. *N* *child* containers each hold
one child and serve it on CPU through llama.cpp. All run on one Docker network.

**Lifecycle.**

1. *Healthy.* Main serves the parent. Children exchange heartbeats with main and can also
   serve requests.
2. *Main down.* Children detect missed heartbeats and continue serving. One child acts as the
   request endpoint.
3. *Main back, empty.* The operator restarts the main container with no weights. It reports
   that it is waiting for shards.
4. *Rebuild.* Children send their codes. Main sums them, checks the hashes in the manifest,
   loads the parent and resumes serving.
5. *Partial rebuild.* If children are also missing, main rebuilds from the rest at reduced
   precision and records which offsets were absent.

**Operator controls.** Only the operator starts containers. Children send shards only to the
registered main service. A signed revoke message stops every node and deletes its shards.

## Alternatives considered

**SVD slices with a shared low-rank backbone.** The original proposal. Storing a rank-*r*
factorization of a *d*×*d* matrix costs 2*dr* values. A 3x per-node memory reduction limits a
node to rank *d*/6, while an exact rebuild needs total rank *d* because LLM weights are
numerically full rank. These two constraints require at least six nodes and no backbone at all.
Interleaved slices also make poor standalone models, since no node holds the top singular
directions, and a lost slice cannot be recovered from the others. Rejected.

**Erasure coding of raw weights.** Exact and storage-efficient, but shards cannot serve
requests. Kept as a complement for bit-exact recovery (see future work).

**Splitting attention heads and FFN neurons across children.** Produces functional children
and an exact rebuild by concatenation, but requires training with structured dropout and
distillation. Deferred to future work in favor of a training-free method.

**Random dithering.** Averaging independently dithered children reduces error only as
1/√*N*. Staggered offsets are strictly better for the same storage.
