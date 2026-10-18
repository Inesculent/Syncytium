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

## Offset phase

All offsets can be shifted by a common phase φ in [0, 1), so child *k* uses (*k* + φ)/*N*.
Applying Hermite's identity to *x* + φ/*N* gives *Q* = Σₖ *q*ₖ = ⌊*N x* + φ⌋, so the rebuild
stays exact, with *ŵ* = *m* + (Δ/*N*)(*Q* + ½ − φ). With φ = 0 the parent is a floor quantizer
on the fine grid; with φ = ½ it is round-to-nearest.

The phase changes which children exist. The child with offset ½ places levels exactly on each
group's minimum and maximum, so it reproduces outliers exactly; the child with offset 0 misses
both by Δ/2. Choosing φ = ½ removes the offset-0 child. With φ = 0, offsets nest across
powers of two (the children for *N* = 4 are a subset of those for *N* = 8), which the sweep
uses to avoid evaluating the same model twice.

## Partial rebuild

With only a subset of children available, each child still bounds the true value: child *k*
places *x* in the interval [*q*ₖ − *k*/*N*, *q*ₖ + 1 − *k*/*N*). The estimate is the midpoint of
the intersection of these intervals. Its worst-case error is half the largest gap between the
available offsets, in units of Δ. With 4 children and one missing, that is Δ/4: half the error
of a single child.

## What each child stores

Each child is one safetensors file:

- Integer codes for every quantized weight, packed densely to *b* bits (least significant bit
  first, so 3-bit codes need no padding per value).
- The shared per-group minimum *m* and step Δ in fp16 (identical in every child). When a grid
  is stored, *m* is rounded down and Δ up to the nearest fp16 values, so every code still fits
  in *b* bits and the rebuild stays exact. Δ grows by at most about 0.1%; phase 1 simulations
  use the unrounded float32 grid.
- Every tensor that is not quantized (norms, Gated DeltaNet convolution and decay parameters),
  unchanged.

Every node also keeps the set's manifest: model id, scheme (*b*, *g*, *N*, φ, rotation seed),
the rotation's signs, the name and shape of every tensor, the SHA-256 of each child's file,
and the SHA-256 of the parent's integer codes. A shard is checked against its hash before
use, and a full rebuild must reproduce the parent hash, which proves it is bit-exact. The
parent hash covers codes in the rotated basis, not the rotation itself, so readers take the
signs from the manifest rather than regenerating them from the seed, which would depend on
PyTorch's random number generator.

A rebuild streams the shards one tensor at a time, so its peak memory is the rebuilt weights
plus a few copies of the largest tensor rather than *N* whole children.

Embeddings are quantized like any other weight. Qwen3.5 uses a 248K-token vocabulary with tied
embeddings, so the embedding table is roughly a quarter of the 2B model and cannot be left in
bf16 if the memory target is to be met.

## Optional shared rotation

Before quantization, each group of *g* weights can be multiplied by an orthogonal *g*×*g*
matrix *R* = *D H* / √*g*, where *H* is a Sylvester Hadamard matrix and *D* a diagonal of
random signs. Children quantize the rotated groups, and weights are recovered by multiplying
back by *R*ᵀ. *R* is fixed by a seed shared across all children (shards store the resulting
signs) and the transform is linear,
so the rebuild identity still holds. Rotation spreads outliers across the group, shrinking its
range and therefore Δ. The per-weight error bounds then hold in the rotated basis; in the
original basis the error of each group is bounded in Euclidean norm by √*g* times the
per-weight bound. Because *R* acts within a group, it needs no power-of-two model dimension. In phase 1 it is simulated by dequantizing in
PyTorch; running rotated children efficiently needs kernel support that stock llama.cpp lacks.

## Optional channel scaling

Before quantization, input channel *c* of a weight can be multiplied by a scale *s*_c and the
dequantized weight divided by it again, as in AWQ: channels that see large activations are
rounded relatively more finely. As with the rotation, every child uses the same scales and
the transform is linear, so the rebuild identity still holds. Channel scaling and the
rotation can be combined; the weight is scaled first and rotated within groups after.

The scales are *s* = *a*^α, normalized so that max(*s*) · min(*s*) = 1, where *a*_c is the
root mean square of input channel *c* on calibration text (64 windows of 512 tokens from the
WikiText-2 train split, never the test split that perplexity is measured on). α is chosen
per weight from 0, 0.05, …, 1 to minimize the activation-weighted squared error
Σ_c E[*x*_c²] Σ_r (*ŵ*_rc − *w*_rc)² of the round-to-nearest child at the children's bit
width. This is a diagonal approximation of the layer's output error, cheaper than AWQ's
search over stored activations. α = 0 is no scaling, so the search never does worse than the
unscaled weight on this objective. The scales do not depend on *N* or φ, so every set of
children at one bit width shares them, and scaled round-to-nearest models serve as an
AWQ-style single-model baseline.

Stored as fp16, the scales add under 0.006 bits per weight over a whole model (at most 1 bit
per weight for the smallest layer), so they are not counted in the memory figures. Shards
cannot store them yet; `syncytium-shard` refuses scaled schemes.

## Memory accounting

Bits per weight for a child are *b* + 32/*g* when *m* and Δ are stored as fp16 per group:

| *b* | *g* | Bits per weight | Reduction vs bf16 |
|----:|----:|----------------:|------------------:|
| 4   | 64  | 4.5             | 3.6x              |
| 3   | 64  | 3.5             | 4.6x              |
| 2   | 64  | 2.5             | 6.4x              |

Reported numbers always include unquantized tensors (kept in bf16). Phase 1 computes them from
parameter counts; phase 2 also measures the shard files on disk.

## Pooled pipeline

Devices too small to hold the model can still serve it together. The parent is split into
stages: stage 0 holds the embedding (tied to the output head), and stages 1 … *S* hold
contiguous blocks of decoder layers. Each stage is held by *N* devices, and device (*s*, *k*)
stores only child *k* of stage *s*, so it needs about 1/(*S* + 1) of one child's memory.

Because the parent's weights are exactly the mean of the children's weights (see "Rebuild"),
and every linear layer is linear in its weights,

  *Ŵ x* = (1/*N*) Σₖ *Ŵ*ₖ *x*.

At every linear layer of the stage, each device multiplies the same input by its own child
and the devices average their outputs before the next nonlinearity. Averaging only the
stage's final output would not work: the norms, activations, attention and Gated DeltaNet
recurrence between linear layers are not linear, so that gives an ensemble of per-child
stages, not the parent. With every device alive the result is the rebuilt parent (up to
floating-point rounding in the averaging), although no device ever holds more than one
child. Activations pass between stages as in any pipeline; stage 0 is visited twice per
token, for the embedding lookup and the output head.

When devices fail, a stage averages its surviving devices instead (the *mean* estimator).
This is the only estimator the devices can compute without gathering each other's codes.
It is slightly less accurate than the interval midpoint of "Partial rebuild": on Gaussian
weights with one of *N* children missing, its mean squared error is 6–30% higher. A stage
fails only when all *N* of its devices are lost, so any *N* − 1 failures are survivable,
and with independent device failure probability *p* the pipeline runs with probability
Π (1 − *p*ᴺ) over its stages.

`syncytium-pool` simulates this: for each state of the device grid it sets every stage's
weights from that stage's surviving children and measures perplexity. Setting the weights to
the children's mean is equivalent to averaging every linear layer's outputs, so the
simulation matches the per-layer protocol above. A rotation or channel scales, shared by
every child, are linear and do not change this: each device dequantizes its own child in the
original basis before multiplying.

## Baselines on the same nodes

Staggered children are compared with two simpler ways of giving each of *N* nodes one shard,
judged by worst-case perplexity when *s* nodes survive and by memory per node:

- *Replicated.* Every node holds the same round-to-nearest model at the children's bit width.
  It costs the same memory as a child, and its quality does not depend on *s*.
- *Coded remainder* (*k*). Every node holds the same base child plus one share of what the
  base lacks from the parent, erasure-coded so that any *k* shares recover it. With child *k*'s
  code *q*ₖ = ⌊*x* + (*k* + φ)/*N*⌋ and the parent's *Q* = ⌊*N x* + φ⌋, writing
  *x* + (*k* + φ)/*N* = *q*ₖ + *f* with *f* in [0, 1) gives *Q* − *N q*ₖ = ⌊*N f*⌋ − *k*, one
  of *N* values. The remainder therefore takes log₂ *N* bits per weight, and each share
  log₂(*N*)/*k*. Fewer than *k* survivors serve the base child; *k* or more rebuild the parent
  exactly. With *k* = 1 every node holds the whole parent, which is also what nested
  (Matryoshka-style) slicing of one stored model amounts to. The base is the child whose
  offset is closest to ½ (round-to-nearest when φ = 0).

Both baselines are computed from conditions the sweep already measures. Neither can pool:
coded shares must be decoded before they can be multiplied, and identical replicas add
nothing by averaging. `syncytium-report` writes the comparison as `baselines.md`.

## Equal memory per node

A node's memory can hold a large model at few bits or a smaller model at more bits, and for
a single model the second is often better. Staggered children are therefore also compared
across model sizes at the weight memory of one child: against the best single model of any
measured size that fits in that memory (bf16 or round-to-nearest at any bit width), with one
node alive (the worst child) and with the fleet (the worst rebuild from *N* − 1 children, and
the rebuild from all *N*). Memory counts codes, scales and unquantized tensors, not the
key/value cache. Parameter counts come from each model's config, built on PyTorch's meta
device, and are checked against the memory each sweep logged. `syncytium-compare` writes the
comparison.

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

**Serving modes while main is down.**

- *Solo.* One child generates the whole response. Child quality at single-machine speed, with
  no traffic between children.
- *Pooled.* The live children run the parent together in lockstep. At every linear layer each
  child multiplies the same input by its own weights, and the results are averaged before the
  next nonlinearity (see "Pooled pipeline"). With every child alive the output is the
  parent's, although no node holds parent weights. The averaging must happen at every layer:
  averaging the children's final outputs gives an ensemble, not the parent. Each layer needs
  about four averaging rounds (the attention or Gated DeltaNet input projections, their output
  projection, the MLP's gate and up projections, and its down projection). The gate and up
  outputs are several times the hidden size and dominate the bytes exchanged per token. Every
  child keeps an identical copy of the key/value cache. If a child fails mid-generation, later layers average the survivors and
  generation continues; cache entries already computed stay valid.
- *Rebuild.* As in steps 4 and 5, after which traffic returns to main.

The request endpoint uses pooled mode when enough children are reachable with low latency,
and solo mode otherwise or for latency-critical requests.

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
