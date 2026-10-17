# Plan

Work proceeds in stages. Each stage ends with passing lint and tests and is committed on its
own. Decisions taken while planning are recorded at the end.

## Phase 1: Perplexity sweep (local GPU)

| Stage | Scope | Location |
|-------|-------|----------|
| 1.1 Quantization core | Per-group grids, staggered child codes, integer rebuild, partial rebuild by interval intersection. Tests for the identity, code ranges and error bounds. | `src/syncytium/quant/` |
| 1.2 Model integration | Load the Qwen3.5 text stack; select tensors to quantize (linear weights and embeddings); swap child or parent weights in place. Tests on a tiny random config. | `src/syncytium/models/` |
| 1.3 Evaluation | WikiText-2 perplexity with fixed windows; memory accounting in bits per weight including unquantized tensors. | `src/syncytium/eval/` |
| 1.4 Sweep runner | TOML configs; resumable JSONL results with config, seed, versions and commit. Offsets for *N* = 2 and 4 are subsets of *N* = 8, so only 8 distinct children are evaluated per setting. Includes an offset-phase ablation, *k*/*N* versus (*k* + ½)/*N* (see below). | `scripts/`, `configs/` |
| 1.5 Shared rotation | Block-diagonal randomized Hadamard transform with block size equal to the group size. Test that the rebuild identity still holds. | `src/syncytium/quant/` |
| 1.6 Pilot and sweep | Report module (summary table with H3/H4 checks, recovery, Pareto and offset figures) and SLURM scripts. 0.8B pilot; fix the usable-child threshold in the hypothesis document; run 2B (locally or on the cluster), and 4B and 9B on the cluster (4B's bf16 weights alone exceed a 10 GB GPU); write-up. | `src/syncytium/report/`, `slurm/`, `docs/results/` |

## Phase 2: Container demo

| Stage | Scope |
|-------|-------|
| 2.1 Shard format | Bit-packed codes in safetensors plus a JSON manifest with hashes; round-trip tests. Done: `syncytium-shard` writes 0.8B children of 0.471 GB each (4-bit, *g* = 32), matching the memory formula exactly; exact rebuild with hash check takes about 13 s on CPU. |
| 2.2 GGUF export | 4-bit children written as llama.cpp Q4_1 blocks (scale Δ, per-block offset *m* + Δ(½ − *k*/*N*)), patched into an F16 GGUF produced by llama.cpp's converter. |
| 2.3 Harness (control plane) | Explicit state machine (healthy, main down, awaiting shards, rebuilding); deterministic leader election among children (lease renewed by heartbeats; lowest live index wins); FastAPI endpoints for health, manifest, shards and generation; rebuild client; HMAC-authenticated operator revoke. |
| 2.4 Docker | Child image (llama.cpp server plus harness), main image (CUDA), compose file, demo script for kill, wipe, restart and rebuild. |
| 2.5 Demo run | Event timeline and outputs before, during and after the outage. |
| 2.6 Agent layer | A LangGraph agent works on multi-step tasks through a router node that sends each call to the main model when healthy, to the best available child or partial rebuild during an outage, and back after the rebuild. Measures task success and step quality before, during and after a failure. |

## Phase 3: Cluster

Qwen3.5 9B and possibly 27B dense, zero-shot evaluations, GPTQ/AWQ reference baselines,
SLURM job scripts. A combined expert-split and staggered-precision study for MoE models is a
candidate follow-up.

## Known risks

- ~~Qwen3.5 checkpoints are packaged as multimodal models; loading only the text stack may
  need a weight-name mapping.~~ Resolved in stage 1.2: `AutoModelForCausalLM` loads the text
  stack directly. The reference PyTorch Gated DeltaNet kernels run a 2048-token window of the
  0.8B model in about 0.2 s, so the optimized kernels are not needed.
- llama.cpp support for Qwen3.5 small models must be confirmed at the start of stage 2.2.
- Local sweeps share a 10 GB GPU with the Windows desktop. The pilot crashed once
  (`CUBLAS_STATUS_EXECUTION_FAILED`) while a memory-heavy job ran alongside it; it resumed from
  its log. Run nothing heavy next to a local sweep.
- llama.cpp has no plain 2- or 3-bit block format, so containerized children are 4-bit.
  Results at 2 and 3 bits come from phase 1 only.

## Observations

- **Child quality depends on the offset** (stage 1.2 smoke test, one 2048-token passage, so
  preliminary). With offset ½ the grid places levels exactly on each group's minimum and
  maximum, so outliers are reproduced exactly; with offset 0 both extremes are off by Δ/2.
  Centering the offsets at (*k* + ½)/*N* keeps the rebuild identity (the fine grid becomes
  round-to-nearest) and avoids the worst-placed child; the sweep tests this.
- **Rotation helps children, not rebuilds** (stage 1.5 check, 0.8B, 16 windows, so
  preliminary). The rotated round-to-nearest child improved at every bit width (4-bit: 14.52
  to 14.29; 3-bit: 28.2 to 24.2), while full rebuilds barely moved. 3-bit children of the
  0.8B model remain far from bf16 (12.89) even with rotation.
- **Unusable children can still rebuild a usable parent.** Four 2-bit children each score a
  perplexity above 10,000, yet their rebuild scores 15.5–15.9. Children carry the parent's
  information even when they cannot serve on their own.
- **Open question: rotation block size.** The rotation currently mixes values only within one
  quantization group. Larger blocks spread outliers further and may help low-bit children
  more; a candidate ablation once the pilot is in.

## Decisions

| Decision | Choice |
|----------|--------|
| Array library | PyTorch is a core dependency; CI installs the CPU wheel. |
| Group sizes | Sweep 32, 64 and 128; the demo uses 32 to match Q4_1 blocks (5 bits per weight at 4 bits, a 3.2x reduction). |
| Thresholds | Set after the 0.8B pilot and before the full sweep. |
| Quantizer baselines | GPTQ/AWQ deferred to phase 3. |
| Control plane versus agent layer | Leader election, rebuild decisions and verification are deterministic code with no model in the loop: they need guarantees, must be testable, and keep the operator's revoke authoritative. LangGraph appears only in the agent layer (2.6), where it chooses which model answers a request using health reported by the control plane; it never decides whether to rebuild or which nodes exist. |
| MoE expert splitting | Not pursued now: well covered by expert-pruning work and cluster-only. Kept as a phase 3 extension combined with staggered precision. |
