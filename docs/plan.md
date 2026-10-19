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
| 1.6 Pilot and sweep | Report module (summary table with H3/H4 checks, recovery, Pareto and offset figures) and SLURM scripts. 0.8B pilot; fix the usable-child threshold in the hypothesis document; run 2B (locally or on the cluster), and 4B and 9B on the cluster (4B's bf16 weights alone exceed a 10 GB GPU); write-up. Done at all four sizes; the write-up is the [phase 1 report](results/phase-1-report.md). | `src/syncytium/report/`, `slurm/`, `docs/results/` |
| 1.7 Pooled pipeline | Stages of the parent held by *N* devices with one child each; a stage averages its live devices' outputs, which with all alive equals the parent exactly. `syncytium-pool` evaluates every single-device loss, every stage on one child, and random loss patterns with their per-stage components, under the mean and midpoint estimators; the summary checks H6b and H6c. Pool grids take `rotation` and `scaling` like the sweep grid. Run at 0.8B (`pool-0.8b`, 4-bit, rotated), then 2B (`pool-2b`, 3-bit, rotated and scaled, the setting a pooled demo would use). Done: H6 holds at both sizes. | `src/syncytium/pool/`, `configs/pool-*.toml` |
| 1.8 Baselines | Staggered children against the same model on every node and against a base child plus an erasure-coded remainder, by surviving nodes and memory per node (see [method](method.md#baselines-on-the-same-nodes)); computed from sweep logs by `syncytium-report`. Model sizes compared at equal memory per node (see [method](method.md#equal-memory-per-node)) by `syncytium-compare`. `configs/subsets-0.8b.toml` measures every survivor count for *N* = 4 at 0.8B, including the two-of-four rebuilds the pilot lacks (about 1.6 h); done. `configs/subsets-4b.toml` repeats it at 4B, at φ = ½, with and without rotation and scaling (209 conditions, about 2 h); done. | `src/syncytium/report/`, `configs/` |
| 1.9 Channel scaling | AWQ-style activation-aware channel scales shared by every child (see [method](method.md#optional-channel-scaling)), searched per weight on calibration text from the train split; a `scaling` option in the sweep grid next to `rotation`. Answers "why not a better quantizer": scaling composes with staggering exactly, while GPTQ-style error feedback does not. `configs/scaling-{0.8b,2b,4b,9b}.toml` compare scaled and unscaled children and single models at each size; done. `configs/seed{1,2}-9b.toml` repeat the 9B 3- and 4-bit settings with two more rotation seeds, to show which differences of a few hundredths are real (about 30 min each); done. | `src/syncytium/quant/`, `src/syncytium/models/`, `configs/` |

## Phase 2: The swarm, in simulation

After phase 1 the project was reframed as a swarm of agents (see the
[swarm pre-registration](preregistration-swarm.md)). Every claim is proven by unit tests,
measured in simulation on the cluster, or modelled from exact counts; real networks and
containers are follow-up work.

| Stage | Scope | Location |
|-------|-------|----------|
| 2.S1 Task harness | GSM8K (8-shot chain of thought) and MMLU (5-shot) for bf16, every child, every fused group and the full group losing each member mid-answer; per-answer logs; the pre-registered checks (H7, H8, H11, H13). Gate A ran at 9B and 4B. Done. | `src/syncytium/tasks/`, `configs/swarm-*.toml` |
| 2.S2 Regrowth | Neighbour bits and an erasure-coded residue (Reed–Solomon over GF(2⁸)) next to the shards; hash-checked regrowth of lost members (H14, H15). Done. | `src/syncytium/quant/regrow.py`, `src/syncytium/shards/` |
| 2.S3 Speed model | Rounds traced from the architecture, drafting agreement measured on the cluster, tokens per second under a stated network model (H9). Done. | `src/syncytium/speed/`, `configs/speed-*.toml` |
| 2.S4 Swarm console | `syncytium-swarm`: real child shards as agents in one process; ask, solo, fuse, kill and regrow. Done. | `src/syncytium/swarm/` |
| 2.S5 Escalation on disagreement | Confirm H8b and H10b on GSM8K train problems at 9B and 4B, with a 6-bit round-to-nearest model as the static splits' large model. Done: H8b holds and H10b fails at both sizes. | `configs/confirm-*.toml`, `src/syncytium/tasks/disagreement.py`, `src/syncytium/speed/elasticity.py` |
| 2.S6 Survival | H12b ([survival pre-registration](preregistration-survival.md)): expected per-request accuracy under device loss, against two cross-routed 6-bit pipelines, no backup, judged at 10–30% failure. Code done; needs the 6-bit model's GSM8K accuracy (`rtn_bits = [6]` in the swarm configs). | `src/syncytium/speed/survival.py` |
| 2.S7 Remaining | H16 (fusion partners, *N* = 16), the regrowth negative control's perplexity. | — |

## Deferred: container demo

Written for the original framing. Stage 2.1 is done and serves the swarm; the rest waits for
follow-up systems work.

| Stage | Scope |
|-------|-------|
| 2.1 Shard format | Bit-packed codes in safetensors plus a JSON manifest with hashes; round-trip tests. Done: `syncytium-shard` writes 0.8B children of 0.471 GB each (4-bit, *g* = 32), matching the memory formula exactly; exact rebuild with hash check takes about 13 s on CPU. The rebuild streams tensors from the shards; a bf16 rebuild of the 0.8B set peaks at 3.8 GB of process memory (9.8 GB when every child was loaded whole). Rotated manifests store the rotation's signs. Not yet: channel scales. They are one fp16 vector per weight, identical in every child, and must be stored with the set and covered by the parent hash before scaled children can be sharded. |
| 2.2 GGUF export | 4-bit children written as llama.cpp Q4_1 blocks (scale Δ, per-block offset *m* + Δ(½ − (*k* + φ)/*N*)), patched into an F16 GGUF produced by llama.cpp's converter. Q4_1 stores the offset in fp16, so GGUF children are close to but not bit-identical with the shard children; rebuilds always use the shards. Q4_1 packs value *j* of a block in the low nibble of byte *j* and value *j* + 16 in its high nibble, unlike the sequential layout of the shards, so codes are repacked rather than copied. Rotated children cannot be exported (no kernel support). Scaled children need their channel scales folded into the operation that produces each layer's input (a norm, or the previous linear layer), since Q4_1 has no per-input-channel divisor; folding into a norm is shared by every child and keeps the rebuild exact, but whether every quantized input in Qwen3.5 has such a predecessor is unchecked. Before export, `configs/demo-2b.toml` measures the demo's scheme in simulation (4-bit, *g* = 32, φ = ½, every survivor count, with and without rotation and scaling). |
| 2.3 Harness (control plane) | Explicit state machine (healthy, main down, awaiting shards, rebuilding); deterministic leader election among children (lease renewed by heartbeats; lowest live index wins); FastAPI endpoints for health, manifest, shards and generation; rebuild client; HMAC-authenticated operator revoke. Every request and event is logged as JSONL with the system measurements listed in the [hypothesis](hypothesis.md#evaluation-protocol) from the first version, not added afterwards. |
| 2.4 Docker | Child image (llama.cpp server for solo serving, a PyTorch CPU worker for pooled serving, and the harness), main image (CUDA), compose file, demo script for kill, wipe, restart and rebuild. Pooled serving averages the children's outputs at every linear layer; a test checks that its logits match the rebuilt parent's within a tolerance for floating-point rounding (averaging bf16 outputs is not bitwise equal to multiplying by the averaged weights). |
| 2.5 Demo run | Event timeline and outputs before, during and after the outage. Solo and pooled serving are compared on the same queries (quality, tokens per second, bytes exchanged per token), including a child killed mid-generation in pooled mode. |
| 2.6 Agent layer | A LangGraph agent works on multi-step tasks through a router node that sends each call to the main model when healthy; during an outage to pooled children when enough are reachable, otherwise to the best available child; and back after the rebuild. Measures task success and step quality before, during and after a failure. |
| 2.7 Pooled containers | The pooled pipeline of stage 1.7 as containers, extending the pooled worker of 2.4 from whole children to stages: one worker per (stage, child) device, each with a memory limit below what the whole model needs, on a local Docker network with no internet access. A stage averages its live workers' outputs; activations pass between stages. Membership comes from an operator-written roster; workers never discover or add devices, and the signed revoke from 2.3 stops every worker. Demo: kill workers one at a time, then whole sets, and record generation quality, tokens per second and bytes moved; check that outputs match the stage 1.7 simulation for the same device states. |

## Phase 3: Cluster

- **27B and H17.** Whether 2-bit children become usable at 27B. No 27B Base exists, so H17
  runs on the post-trained series 0.8B–27B (`configs/post-*.toml`); queued.
- **GPTQ** as a single-model reference baseline. AWQ-style scaling is covered by stage 1.9.
- **Instruction-tuned agents.**

The 9B sweeps and the SLURM scripts are done. A combined expert-split and staggered-precision
study for MoE models is a candidate follow-up.

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
  Results at 2 and 3 bits come from phase 1 only, unless pooled workers serve them in PyTorch.
- Some cluster nodes fail CUDA initialization (error 101). Both early 2B attempts and the
  first scaling runs died this way after logging a start with `device: cpu`.
  `slurm/sweep.sbatch` now stops at once with `CUDA unavailable on this node`. The known bad
  node is `evc26`; submit with `--exclude=evc26`.
- ~~A command added after the cluster's environment was installed does not exist there~~:
  the first pool runs failed with `syncytium-pool: command not found`, because an editable
  install picks up new code but not new commands. `slurm/sweep.sbatch` now starts task and
  speed runs as modules (`python -m …`), which needs no reinstall.
- The swarm's case rests on survival. On a reliable network, a 6-bit model pipelined over
  the same devices is several times faster than fusing at similar accuracy (H10b is
  registered as expected to fail), so H12 (survival under churn) decides whether the
  swarm has an advantage at all.

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
- **3-bit children become usable from 4B** (stage 1.6, 4B and 9B sweeps). The worst 3-bit
  child is 1.31–1.40x bf16 at 4B and 1.32–1.41x at 9B, against 1.97–2.70x at 0.8B, so every
  3-bit setting passes H2 at both sizes; the ratio does not improve further from 4B to 9B.
  4-bit children are 1.05–1.06x at both. 2-bit children stay unusable, but eight of them
  rebuild to within 0.19 of bf16 at 4B and 0.15 at 9B; four fall short of H3 (0.54–0.79).
  H4 holds in every setting at every size. At 2B, 4-bit children pass (1.08–1.10x) and 3-bit
  children narrowly fail (1.54–1.78x), so the crossing lies between 2B and 4B.
- **Combining matters most at 3 bits.** At 9B (N = 4, rotation, φ = ½), combining takes the
  worst 3-bit child from 9.27 to 7.08 (bf16 6.98), but the worst 4-bit child only from 7.34 to
  7.00. With three of four alive, 3-bit staggered children (7.25) beat 4-bit round-to-nearest
  on every node (7.30), which costs one more bit per node; the same holds at 4B (8.34 against
  8.47).
- **The fleet beats any single model that fits on one node** (stage 1.8, 0.8B, 4B and 9B,
  N = 4). In every 3- and 4-bit setting, any three of four children rebuild a better model
  than the best single model of any measured size that fits in one child's memory. At 9B with
  3-bit children (3.9 GB per node), three of four give 7.25 and all four 7.08, against 8.07
  for the best single fit (4B at 6 bits). The same holds at 2B. One node alone is worse than
  the best fit: by 0.75–1.2 perplexity for 3-bit children at 4B (where 2B at 7 bits, 10.06, is
  the best fit) and 1.2–1.8 at 9B, but at most 0.1 for 4-bit children at 4B and 9B.
- **Neither baseline dominates** (stage 1.8, 0.8B pilot, N = 4; the same pattern at 4B and 9B). With one node left, the
  same 4-bit model on every node is slightly better than the worst staggered child (14.66
  against 14.85 with rotation and φ = ½). With three of four, staggered children score 13.34,
  and the replicated model needs about 2 more bits per node to match them; the coded
  remainder rebuilds the parent (13.20) from any three for 0.67 more bits per node. With all
  four, staggered matches the coded scheme at no extra memory. With two of four (from
  `subsets-0.8b`), staggered children score 13.77 in the worst case against 14.66 for
  replication, and the coded scheme with *k* = 2 needs one more bit per node.
- **Every lost child costs quality, and which ones are lost matters** (stage 1.8,
  `subsets-0.8b`). In all twelve settings, any *k* + 1 of four children rebuild a better
  model than any *k*. Two children with offsets ½ apart rebuild far better than two
  adjacent ones (3-bit, φ = ½, rotation: 14.90–15.02 against 16.47–16.96), as the
  partial-rebuild bound predicts. For the demo roster this means each failure domain should
  hold evenly spaced offsets (children 0 and 2 together, 1 and 3 together). The overlap with
  the pilot reproduces every perplexity below 100 within 0.005 across GPUs.
- **Channel scaling composes with staggering and removes most of the single-node penalty**
  (stage 1.9, N = 4, φ = ½). Scaling improves every child, rebuild and single model. With
  rotation, the worst 3-bit child goes from 1.33x to 1.13x bf16 at 9B and from 1.56x to 1.30x
  at 2B, which now passes H2; four 2-bit children rebuild within H3's bar at 4B (0.44) and 9B
  (0.27). With scaling and rotation, the worst child is within 0.05 of the scaled RTN model
  of the same size at 3 and 4 bits at 4B and 9B (9B 4-bit: 7.159 against 7.157), and the
  children of a set become nearly interchangeable (9B 3-bit: 7.86–7.90). Without rotation the
  scaled 3-bit children still trail by about 0.2. At equal memory, any three of four scaled
  children beat the best single model in every setting, 2-bit included.
- **Fusion recovers the parent's task accuracy** (Gate A). Four fused 3-bit children solve
  86.1% of GSM8K at 9B (bf16 86.2%; lone children 76.8–81.8%) and 82.9% at 4B (bf16 82.6%;
  68.0–74.2%). Children whose perplexities differ by hundredths differ by 5–6 points on
  GSM8K.
- **Entropy is a weak escalation signal; disagreement is a strong one.** Escalating the
  30% most uncertain answers recovers 36–42% of the fusion gain (H8 fails). Escalating when
  two lone agents give different answers recovered 94–104% while escalating 26–37%. That was
  found after the fact, then confirmed on fresh problems: 92–110% while escalating 23–35%
  (H8b holds at both sizes).
- **The 3-bit "within 0.05 of RTN" result was one seed's luck.** Across three rotation seeds
  at 9B the gap is 0.05–0.09 at 3 bits, though still at most 0.014 at 4 bits.
- **Open question: rotation block size.** The rotation currently mixes values only within one
  quantization group. Larger blocks spread outliers further and may help low-bit children
  more; a candidate ablation once the pilot is in.

## Decisions

| Decision | Choice |
|----------|--------|
| Array library | PyTorch is a core dependency; CI installs the CPU wheel. |
| Group sizes | Sweep 32, 64 and 128; the demo uses 32 to match Q4_1 blocks (5 bits per weight at 4 bits, a 3.2x reduction). |
| Thresholds | Set after the 0.8B pilot and before the full sweep: H2 needs the worst child within 1.5x of bf16 perplexity; H3 stays at 0.5; H4 applies for *N* ≥ 3 (see [hypothesis](hypothesis.md#what-results-would-mean)). |
| Quantizer baselines | AWQ-style channel scaling added in stage 1.9, because it composes with staggering exactly. GPTQ-style error feedback does not compose and stays a phase 3 single-model reference. |
| Control plane versus agent layer | Leader election, rebuild decisions and verification are deterministic code with no model in the loop: they need guarantees, must be testable, and keep the operator's revoke authoritative. LangGraph appears only in the agent layer (2.6), where it chooses which model answers a request using health reported by the control plane; it never decides whether to rebuild or which nodes exist. |
| Swarm reframe | After phase 1 the goal became a swarm that survives, not a protected single model: lone agents, fusion on demand, exact regrowth. The original hypotheses stand; H5 is superseded. |
| Simulation only | Networks are modelled, not measured; the paper shows the swarm is possible, and real systems are follow-up work. |
| Pre-registration | Every new hypothesis and threshold is committed before its data exists; clarifications are dated and kept; failures are reported, and a failed hypothesis's registered consequence stands (H8 failing dropped H10). |
| Models and tasks | Base checkpoints, as in phase 1; GSM8K primary, MMLU exploratory; debugging only on the GSM8K train and MMLU dev splits. |
| Fusion estimator | The mean of the live children, what devices compute by averaging outputs; the midpoint was within 0.005 in `pool-2b`. |
| Regrowth | Exact only: neighbour bits for small swarms, the coded residue for several simultaneous losses. Guessing degrades the set and drifts. |
| MoE expert splitting | Not pursued now: well covered by expert-pruning work and cluster-only. Kept as a phase 3 extension combined with staggered precision. |
