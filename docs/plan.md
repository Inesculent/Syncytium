# Plan

Work proceeds in stages. Each stage ends with passing lint and tests and is committed on its
own. Decisions taken while planning are recorded at the end.

## Phase 1: Perplexity sweep (local GPU)

| Stage | Scope | Location |
|-------|-------|----------|
| 1.1 Quantization core | Per-group grids, staggered child codes, integer rebuild, partial rebuild by interval intersection. Tests for the identity, code ranges and error bounds. | `src/holograph/quant/` |
| 1.2 Model integration | Load the Qwen3.5 text stack; select tensors to quantize (linear weights and embeddings); swap child or parent weights in place. Tests on a tiny random config. | `src/holograph/models/` |
| 1.3 Evaluation | WikiText-2 perplexity with fixed windows; memory accounting in bits per weight including unquantized tensors. | `src/holograph/eval/` |
| 1.4 Sweep runner | TOML configs; resumable JSONL results with config, seed, versions and commit. Offsets for *N* = 2 and 4 are subsets of *N* = 8, so only 8 distinct children are evaluated per setting. | `scripts/`, `configs/` |
| 1.5 Shared rotation | Block-diagonal randomized Hadamard transform with block size equal to the group size. Test that the rebuild identity still holds. | `src/holograph/quant/` |
| 1.6 Pilot and sweep | 0.8B pilot; fix the usable-child threshold in the hypothesis document; run 2B and 4B; plots and write-up. | `docs/results/` |

## Phase 2: Container demo

| Stage | Scope |
|-------|-------|
| 2.1 Shard format | Bit-packed codes in safetensors plus a JSON manifest with hashes; round-trip tests. |
| 2.2 GGUF export | 4-bit children written as llama.cpp Q4_1 blocks (scale Δ, per-block offset *m* + Δ(½ − *k*/*N*)), patched into an F16 GGUF produced by llama.cpp's converter. |
| 2.3 Harness | Explicit state machine (healthy, main down, awaiting shards, rebuilding); FastAPI endpoints for health, manifest, shards and generation; heartbeats; rebuild client; HMAC-authenticated operator revoke. |
| 2.4 Docker | Child image (llama.cpp server plus harness), main image (CUDA), compose file, demo script for kill, wipe, restart and rebuild. |
| 2.5 Demo run | Event timeline and outputs before, during and after the outage. |

## Phase 3: Cluster

Qwen3.5 9B and possibly 27B dense, zero-shot evaluations, GPTQ/AWQ reference baselines,
SLURM job scripts. A combined expert-split and staggered-precision study for MoE models is a
candidate follow-up.

## Known risks

- Qwen3.5 checkpoints are packaged as multimodal models; loading only the text stack may need
  a weight-name mapping. Checked at the start of stage 1.2.
- llama.cpp support for Qwen3.5 small models must be confirmed at the start of stage 2.2.
- llama.cpp has no plain 2- or 3-bit block format, so containerized children are 4-bit.
  Results at 2 and 3 bits come from phase 1 only.

## Decisions

| Decision | Choice |
|----------|--------|
| Array library | PyTorch is a core dependency; CI installs the CPU wheel. |
| Group sizes | Sweep 32, 64 and 128; the demo uses 32 to match Q4_1 blocks (5 bits per weight at 4 bits, a 3.2x reduction). |
| Thresholds | Set after the 0.8B pilot and before the full sweep. |
| Quantizer baselines | GPTQ/AWQ deferred to phase 3. |
| MoE expert splitting | Not pursued now: well covered by expert-pruning work and cluster-only. Kept as a phase 3 extension combined with staggered precision. |
