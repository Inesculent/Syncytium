# Swarm Report: Fusion, Survival and Regrowth

*Draft, 2026-10-18. Covers Gate A of the [swarm pre-registration](../preregistration-swarm.md)
at 9B and 4B (H7, H8, H11, H13), the speed model (H9), exact regrowth (H14, H15), and the
follow-up runs registered in phase 1 (pooled pipeline H6, rotation seeds, every survivor count
at 4B, the demo's scheme), and the confirmation of the
[disagreement pre-registration](../preregistration-disagreement.md) (H8b, H10b) on fresh
problems.*

## Summary

- **Fused children answer like the parent.** Four fused 3-bit children solve 86.1% of GSM8K
  at 9B (bf16: 86.2%) and 82.9% at 4B (bf16: 82.6%). The lone children solve 76.8–81.8% and
  68.0–74.2%. Fusing recovers 98% and 102% of the gap, far beyond H7's bar.
- **Every lost member costs a predictable step, and a member lost mid-answer costs nothing.**
  Mean accuracy rises with each member at both sizes (H11). When a fused group loses a member
  halfway through an answer and the survivors finish it from the group's cache, accuracy is
  slightly *higher* than the survivors answering from scratch (H13).
- **A lost member can be regrown exactly from the survivors.** One stored bit per ambiguous
  code (2/*N* bits per weight per child), or an erasure-coded residue for many losses at once,
  regrows every lost member with a hash match and no drift (H14, H15). For 100 members,
  surviving half of them lost at once costs 0.14 bits per weight per child, on disk only.
- **Two agents disagreeing tells the swarm when to fuse; one agent's entropy does not.**
  Escalating a lone agent's 30% most uncertain answers recovers 36% (9B) and 42% (4B) of the
  fusion gain, against 50% required (H8 fails, so H10 is dropped as registered). Escalating
  when two lone agents give different answers, found afterwards and then confirmed on 1,319
  fresh problems, recovers 107–110% (9B) and 92–100% (4B) of the gain while escalating only
  23–24% and 34–35% of problems (H8b passes).
- **Fusion is survivable, not efficient.** Under the pre-registered network model, four fused
  devices reach 35% of one solo device's speed at 9B (H9 passes) but 23% at 4B (H9 fails). A
  fused token costs about six times the device time of a 6-bit model pipelined over the same
  four devices. What the swarm offers that a pipeline does not is that any surviving member
  keeps answering, and lost members come back exactly.
- **The phase 1 follow-ups hold, with one correction.** Both pooled pipelines pass H6.
  Survivor ordering and the placement rule hold at 4B in all twelve settings. But the
  phase 1 claim that a scaled, rotated child is within 0.05 of round-to-nearest fails at
  3 bits across rotation seeds (0.05–0.09); it holds at 4 bits (at most 0.014).

## What was run

| Item | Setting |
|------|---------|
| Models | Qwen3.5 Base at 9B and 4B (Gate A, H9); 0.8B and 2B (pooled pipeline) |
| Children | 3-bit, *g* = 64, *N* = 4, φ = ½, rotated (seed 0) and channel-scaled; groups smaller than four use the mean estimator |
| Tasks | GSM8K test (1,319 problems, 8-shot chain of thought, greedy, last number compared); MMLU test (14,042 questions, 5-shot, by likelihood, exploratory) |
| Conditions | bf16, each child, every group of 2–4 children, the full group losing each member halfway through its answers |
| Speed | Rounds traced from each architecture; drafting agreement on 16 × 2,048 WikiText-2 test tokens; the network model fixed before measuring |
| Hardware | H100 PCIe; every run from a clean commit containing the pre-registration it reports |

All pre-registration texts, clarifications and the estimator decision were committed before
the runs they govern started.

## Results by hypothesis

### H7: fusion changes answers: holds

| | 9B | 4B |
|---|---:|---:|
| bf16 | 86.2% | 82.6% |
| Worst lone child | 76.8% | 68.0% |
| Four fused | **86.1%** | **82.9%** |
| Gain over the worst child | +9.2 points | +14.9 points |
| Share of the gap to bf16 | 98% | 102% |
| McNemar *p* | 5 × 10⁻¹⁸ | 5 × 10⁻³² |

The lone children differ from each other by 5–6 points on GSM8K although their WikiText-2
perplexities differ by a few hundredths: task-level differences between children are far
larger than perplexity suggests.

### H8: a lone agent knows when to ask: fails

Share of the fusion gain recovered by escalating the answers with the highest mean token
entropy, for the worst child (the registered test):

| Escalated | 10% | 20% | 30% | 50% |
|---|---:|---:|---:|---:|
| 9B (child 0) | 21% | 22% | **36%** | 52% |
| 4B (child 1) | 19% | 33% | **42%** | 60% |

Random escalation recovers about as much as it escalates. Entropy beats chance, but not by the
registered margin (50% at 30%). The other children, reported for reference, recover 41–55%
(9B) and 47–56% (4B). As registered, **H10 is dropped**.

### H9: fused serving is fast enough: holds at 9B, fails at 4B

| | 9B | 4B |
|---|---:|---:|
| Rounds per fused token | 130 | 130 |
| Values each device sends per token | 1.68 M (3.36 MB) | 1.38 M (2.77 MB) |
| Drafter agreement with the group | 82.7–83.2% | 79.0–79.5% |
| Solo tokens/s at the pass point | 25.5 | 54.3 |
| Fused tokens/s (best draft length) | 8.9 (*k* = 2) | 12.4 (*k* = 1) |
| Fused / solo (bar: 25%) | **35%** ✅ | **23%** ❌ |

The pass point is 0.2 ms per step, 1 Gbit/s and 100 GB/s of device memory bandwidth. A smaller
child is read faster, so communication weighs more and the ratio falls. The full grids are in
[speed-9b](speed-9b/summary.md) and [speed-4b](speed-4b/summary.md). At 9B:
- the ratio rises to 60–85% on a 10 Gbit/s link with at most 0.2 ms latency and 50–100 GB/s
  devices;
- it falls to 3–28% on fast devices (1,000 GB/s), where reading weights is cheap and the
  rounds dominate.

Drafting helps only where communication dominates, because a drafted token costs a whole solo
pass. The best draft length is 0–2 for 50–100 GB/s devices at up to 0.2 ms, and reaches 7 at
1 ms on fast devices.

### H11: graceful task-level degradation: holds

Mean GSM8K accuracy over the groups of each size:

| Members | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| 9B | 79.8% | 84.1% | 85.0% | 86.1% |
| 4B | 71.0% | 79.1% | 81.4% | 82.9% |

The worst group of three beats the best lone child at both sizes (9B: 84.3% against 81.8%;
4B: 80.4% against 74.2%).

### H13: surviving a member's loss mid-answer: holds

| | 9B | 4B |
|---|---:|---:|
| Mean difference, dropped mid-answer minus survivors from scratch | +0.9 points | +1.4 points |
| One-sided 95% lower bound (margin −2) | +0.18 | +0.58 |

Losing a member halfway costs nothing; the half written by the full group helps the survivors
finish. A real system can keep a fused group's cache when a member drops.

### H14 and H15: exact regrowth without drift: hold

Implemented in `syncytium.quant.regrow` and `syncytium.shards.regrow` and tested by 215 unit
tests over *N* = 4, 8 and 16, 2–4 bits and both phases.

- Every allowed loss pattern regrows children whose shard files match their recorded
  SHA-256.
- Ten rounds of losing and regrowing every member, and ten rounds of losing *t* members at
  once, leave the parent hash unchanged.

Measured storage per child:

| | *N* = 4 | *N* = 8 | *N* = 16 |
|---|---:|---:|---:|
| Neighbour bits (any losses with no two adjacent) | 0.50 | 0.25 | 0.125 |
| Coded residue, *t* = 1 | 0.67 | 0.43 | 0.27 |
| Coded residue, *t* = *N*/2 | 1.00 | 0.75 | 0.50 |

At *N* = 100 the coded residue costs ⌈log₂ 100⌉/(100 − *t*) bits per weight per child: 0.14
for *t* = 50.

Guessing instead (the negative control) recovers a usable member but not the set. At
*N* = 4, a quarter of the ambiguous codes come out wrong, the set's rebuild error grows 4x, and
five rounds of guessing grow it more than 100x. Self-healing has to be exact.

## Phase 1 follow-ups

**H6, pooled pipeline: holds at 0.8B and 2B** ([pool-0.8b](pool-0.8b/summary.md),
[pool-2b](pool-2b/summary.md)). Five stages of four devices each:

| | 0.8B, 4-bit | 2B, 3-bit scaled |
|---|---:|---:|
| bf16 / full pool | 13.125 / 13.197 | 10.052 / 10.195 |
| H6b: worst single device lost, gap to bf16 (bar 0.5) | 0.124 ✅ | 0.222 ✅ |
| H6c: R² of stage-wise prediction, mean estimator (bar 0.9) | 1.000 ✅ | 0.999 ✅ |
| Measured / predicted log-perplexity increase | 1.005 | 1.003 |

A single device's loss costs at most 0.22 perplexity. The cost of any loss pattern is
predicted almost exactly by adding each stage's own cost, so a controller can report the
quality of any device state from a small table. The midpoint estimator beats the mean by only
0.003 (0.8B) and 0.005 (2B) with one device lost, so Gate A used the mean.

**Every survivor count at 4B: both registered criteria hold in all twelve settings**
([subsets-4b](subsets-4b/summary.md)). Any *k* + 1 of four children rebuild a better model
than any *k*. Every pair with offsets ½ apart beats every adjacent pair. 3-bit, rotated and
scaled:

| Children alive | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| Perplexity | 9.62–9.67 | 8.39–8.68 | 8.25–8.27 | 8.14 |
| Pairs | | opposite 8.39–8.40, adjacent 8.63–8.68 | | |

The placement rule (each failure domain holds evenly spaced offsets) now rests on a size
where 3-bit children are usable. On GSM8K at 4B, *exploratory*, the opposite pairs (80.7%,
81.0%) beat every adjacent pair (76.6–79.0%); at 9B the pattern is mixed (84.8% against
82.2–85.4%).

**Rotation seeds at 9B: the "within 0.05" claim fails at 3 bits**
([seed1-9b](seed1-9b/summary.md), [seed2-9b](seed2-9b/summary.md)). Worst scaled, rotated child
minus the round-to-nearest model with the same seed:

| Seed | 0 | 1 | 2 |
|---|---:|---:|---:|
| 3-bit | +0.048 | **+0.055** | **+0.093** |
| 4-bit | +0.002 | +0.014 | +0.013 |

Across seeds, scaled rebuilds move by at most 0.006, single children by about 0.02 and
round-to-nearest models by about 0.03. At 9B, differences between single models below about
0.03 cannot be ranked.

**The demo's scheme at 2B** ([demo-2b](demo-2b/summary.md); no threshold). 4-bit children
with *g* = 32, as Q4_1 stores them. Every survivor count keeps the strict ordering in all
four settings:

| Unrotated (llama.cpp) | 1 child | 2 | 3 | 4 | RTN 4-bit |
|---|---:|---:|---:|---:|---:|
| Unscaled | 10.66–10.75 | 10.20–10.34 | 10.14–10.16 | 10.09 | 10.67 |
| Scaled | 10.43–10.53 | 10.15–10.25 | 10.11 | 10.07 | 10.44 |

bf16 is 10.05.

## Escalation on disagreement: confirmed (H8b), but not efficient (H10b)

On GSM8K train problems 64–1,382, which Gate A never used, with the setting unchanged
([confirm-9b](confirm-9b/summary.md), [confirm-4b](confirm-4b/summary.md)):

| | 9B | 4B |
|---|---:|---:|
| bf16 / 6-bit model | 87.9% / 88.0% | 84.2% / 84.2% |
| Full group / worst lone child | 87.6% / 79.7% | 82.9% / 66.4% |
| **H8b**: escalated (bar 50%) | 23–24% ✅ | 34–35% ✅ |
| **H8b**: gain recovered (bar 75%) | 107–110% ✅ | 92–100% ✅ |
| Random escalation at the same rate would recover | 23–24% | 34–35% |

Every partner passes at both sizes, so the swarm can tell when to fuse: two members answer,
and the group is consulted only when they disagree. At 9B the policy even beats the full
group alone (88.2–88.5% against 87.6%): on the problems where the two members agree, they
are right slightly more often than the group is.

H10b compares the swarm using this router with static splits of the same four devices, in
problems per second per answer token. It **fails at both sizes**, as registered:

| | 9B: accuracy | throughput | 4B: accuracy | throughput |
|---|---:|---:|---:|---:|
| Swarm | 88.3% | 21.9 | 82.2% | 26.9 |
| Split A: two agents + 6-bit on two devices | 88.5% | 25.5 | 82.9% | 54.3 |
| Split B: 6-bit on four devices | 88.0% | 54.4 | 84.2% | 114.6 |
| Floor: four lone agents | 81.9% | 102.1 | 72.1% | 217.2 |

A static split with the same router is at least as accurate and faster: escalated problems
cost far less on a 6-bit pipeline than on four fused devices. The 6-bit model also matched
bf16 on these problems, within 0.1 point at both sizes.

## Exploratory results

**MMLU.** The fusion gain carries over to knowledge questions, though smaller than on
reasoning:

| | 9B | 4B |
|---|---:|---:|
| Children | 68.7–70.0% | 65.7–70.1% |
| Fused | 70.8% | 72.8% |
| bf16 | 71.9% | 73.0% |

**Escalation on disagreement, as first found.** Found after H8's result, on the same test
answers, and confirmed above on fresh problems. The worst child asks; a partner answers the same
problem; the problem goes to the full group when their final numbers differ.

| | 9B | 4B |
|---|---:|---:|
| Escalated (each of the three partners) | 26–27% | 36–37% |
| Gain recovered | 103–104% | 94–98% |
| Entropy at 30% (registered H8) | 36% | 42% |

It needs agents that differ: identical replicas decoding greedily always agree, so the signal
exists only because staggered children are diverse by construction. The cost is a second
answer for every problem.

## What can be concluded

1. **The swarm works as a model.** Fusing staggered children reproduces the parent's task
   accuracy at both sizes. The quality ladder holds at task level as it does in perplexity,
   and a fused group survives a member's loss mid-answer.
2. **The swarm heals exactly.** Lost members come back bit for bit from the survivors at a
   small storage cost that shrinks as the swarm grows. Guessing does not work.
3. **The swarm is not efficient.** On a reliable network, a 6-bit model pipelined over the
   same devices matches its accuracy at several times its throughput (see the
   [disagreement pre-registration](../preregistration-disagreement.md#known-when-fixing)).
   Fusing is slow because every device reads a whole child for every token, and devices
   exchange megabytes per token.
4. **So the claim is survival.** A pipeline stops when any stage is lost. A replica set gains
   nothing from extra members. Erasure-coded shards cannot answer until decoded. The swarm
   answers with any surviving member, gets better with every member present, and regrows the
   lost ones. Quantifying that against the alternatives under churn (H12) is the comparison
   the project now depends on.
5. **The swarm can route itself.** Entropy fails as an escalation signal, but two members
   disagreeing finds nearly every problem worth fusing for, on fresh problems at both sizes.
   It works because staggered children differ; identical replicas would always agree.
6. **What cannot be concluded yet:**
   - H12b, survival under device loss (registered; needs the 6-bit model on the test problems);
   - H16, choosing fusion partners in larger swarms;
   - H17, 27B;
   - downstream tasks beyond GSM8K and MMLU;
   - instruction-tuned agents;
   - any measurement on real networks or devices.

## Status of every hypothesis

| Hypothesis | Outcome | Where |
|---|---|---|
| H1–H4 | Hold | [Phase 1 report](phase-1-report.md) |
| H5 (wipe and rebuild the main container) | Superseded by H13–H15 | [Swarm pre-registration](../preregistration-swarm.md) |
| H6b, H6c (pooled pipeline) | Hold at 0.8B and 2B | Above |
| Seed robustness ("within 0.05") | Holds at 4 bits, fails at 3 bits | Above |
| Survivor ordering and pairs at 4B | Hold | Above |
| H7 (fusion changes answers) | Holds at 9B and 4B | Above |
| H8 (entropy escalation) | Fails at 9B and 4B | Above |
| H9 (fused speed) | Holds at 9B, fails at 4B | Above |
| H10 (elasticity) | Dropped, as registered after H8 failed | — |
| H11 (graceful degradation) | Holds at 9B and 4B | Above |
| H12 (survival under churn) | Superseded by the confirmatory H12b; awaiting the 6-bit model's accuracy | [Survival pre-registration](../preregistration-survival.md) |
| H13 (loss mid-answer) | Holds at 9B and 4B | Above |
| H14, H15 (exact regrowth, no drift) | Hold | Above |
| H16 (fusion partners, *N* = 16) | Not run | — |
| H17 (27B) | Post-trained series 0.8B–27B queued (no 27B Base exists) | [Swarm pre-registration](../preregistration-swarm.md#changes-after-fixing) |
| H8b (escalation on disagreement) | Holds at 9B and 4B, on fresh problems | Above |
| H10b (elasticity with that router) | Fails at 9B and 4B, as expected when registered | Above |

## Data and figures

| Result | Report |
|--------|--------|
| Gate A, 9B and 4B | [swarm-9b](swarm-9b/summary.md), [swarm-4b](swarm-4b/summary.md) |
| H9 | [speed-9b](speed-9b/summary.md), [speed-4b](speed-4b/summary.md) |
| Pooled pipeline | [pool-0.8b](pool-0.8b/summary.md), [pool-2b](pool-2b/summary.md) |
| Every survivor count at 4B | [subsets-4b](subsets-4b/summary.md), [baselines](subsets-4b/baselines.md) |
| Rotation seeds at 9B | [seed1-9b](seed1-9b/summary.md), [seed2-9b](seed2-9b/summary.md) |
| Demo scheme at 2B | [demo-2b](demo-2b/summary.md) |
| H8b and H10b | [confirm-9b](confirm-9b/summary.md), [confirm-4b](confirm-4b/summary.md) |

Raw outputs are in `results/<name>/`. Task runs also keep every answer in
`results/<name>/items/<task>/<condition>.jsonl`, with the config, commit and package versions
of each run in `runs.jsonl`.
