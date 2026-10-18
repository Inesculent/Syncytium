# Pre-registration: The Swarm Hypotheses (H7–H17)

*Proposed on 2026-10-18. Fixed by the commit that adds this file, before any GSM8K or MMLU
result exists, including a bf16 sanity check. Nothing here is changed after a result it
governs has been seen; any later change is recorded at the end with its reason. H1–H6 stay as
fixed in [hypothesis](hypothesis.md).*

## Goal

A **syncytium of low-bit agents**. Each device holds one staggered child and runs its own agent.
Any group of devices can **fuse** into a higher-precision version of the same model on demand,
with no extra memory and no weights to load, and split again afterwards. A lost member is
**regrown** from the survivors, and the swarm returns to full health with a hash match. The
swarm degrades gracefully under failure and stops on operator command.

**Scope.** The study shows the swarm is possible, not every application of it. Every claim is
either proven (exact identities, checked by unit tests), measured in simulation (the phase 1
harness on the cluster, which swaps dequantized weights into one model), or modelled from
exact counts under stated assumptions (communication). Real networks, containers and
time to recover are follow-up systems work.

The study has to deliver three results, each measured against equal-hardware alternatives:

1. **Performance.** Fusing makes the swarm better at tasks, and the swarm uses its hardware
   better than a fixed split into small and large models.
2. **Robustness.** Task quality degrades gracefully with lost devices, and a fused group
   survives losing a member mid-generation.
3. **Self-healing.** Survivors regrow a lost member exactly, without any node holding
   parent-precision weights, and repeated losses cause no drift.

## Change from the original goal

| | Original | Swarm |
|---|---|---|
| Question | Can shards be functional models that rebuild the parent? | What can a swarm of functional shards do? |
| Motivation | Protect a deployed model | Elastic compute that survives failures |
| Layout | Main container plus children; staged pool (H6) | One whole child per device; no main |
| Main configuration | 4-bit Q4_1 (llama.cpp) | 3-bit scaled and rotated, *N* = 4 (PyTorch workers) |
| Main metric | Perplexity in each failure state | Task accuracy against compute, under failure |

H1–H4 and every phase 1 result still stand: they show the mechanism works. H5 (wipe and
rebuild the main container) is superseded by H13–H15, because the swarm has no main
container. H6 is reported under its original pre-registration; the staged layout it tests is
no longer the central one. The other documents are rewritten for the swarm only if Gate A
passes.

## Protocol

**Setting.** Unless stated otherwise: Qwen3.5 Base at 9B (with 4B as a second size), 3-bit
children, *g* = 64, *N* = 4, φ = ½, rotation and channel scaling on. These are the Base
checkpoints of phase 1. An "agent" is one running model instance; instruction-tuned models
belong to follow-up work.

**Fused groups.** A group of all *N* children is the rebuilt parent. A group of *s* < *N*
children uses the **mean** estimator (the average of the survivors' weights), unless
`pool-2b` shows the [midpoint estimator](method.md#partial-rebuild) beating the mean by more
than **0.05** perplexity with one device lost, averaged over which device is lost; then it
uses the midpoint. Devices can compute
either: the midpoint is the survivors' mean plus a known shift per group. The estimator not
chosen is reported as secondary.

**Tasks.**
- *Primary: GSM8K.* All 1,319 test problems, the standard 8-shot chain-of-thought prompt,
  greedy decoding, at most 256 new tokens, the final number in the answer compared exactly.
- *Secondary: MMLU, 5-shot,* scored by likelihood. Exploratory, with no threshold: it checks
  whether a fusion gain carries over beyond reasoning.

**Test data stays unseen.** All debugging and smoke tests use the GSM8K *train* split and the
MMLU *dev* split. The test splits are first run by the configuration fixed here.

## Performance

**H7. Fusion changes answers.** Four fused children recover at least **50% of the GSM8K gap**
between the worst lone child and bf16, **and** gain at least **3 points** over the worst lone
child, with a paired McNemar test giving *p* < 0.01. The relative bar keeps the test fair
whatever the ceiling turns out to be (in perplexity, fusion recovers about 95% of the gap at
9B). A gap under 3 points means precision is not the bottleneck, which counts as a failure.
*If it fails:* the fusion headline is dropped and the project returns to the original
protection framing. Gate A.

**H8. A lone agent knows when to ask.** Using the lone agent's mean token entropy over its own
answer as the signal (other signals are exploratory), escalating at most **30%** of problems
recovers at least **50%** of the H7 gain. Escalating at random recovers about as much as the
fraction escalated, so 50% at 30% means the signal clearly beats chance; random escalation is
reported alongside as the control. Escalation is per problem. Per-step escalation is
exploratory, because the attention cache differs between solo and fused modes.
*If it fails:* fusion works but cannot be triggered cheaply. Report the escalation curve and
drop H10. Gate A.

**H9. Fused serving is fast enough, under a stated network model.** Count exactly, from the
architecture, the synchronization rounds and bytes each device sends per fused token (one
round before every nonlinearity). Measure the speculative-decoding acceptance rate in
simulation, with the requesting agent drafting and the fused group checking. Convert both to
tokens per second under a stated network model, with solo time per token taken from the
child's size and the device's memory bandwidth. Passes if four fused devices reach at least
**25%** of one solo device's tokens per second at **0.2 ms per round, 1 Gbit/s and 100 GB/s**
of device memory bandwidth (a wired LAN and an edge-class device; faster devices make
communication relatively more expensive). Also reported over a grid: latency 0.05, 0.2 and
1 ms; 1 and 10 Gbit/s; 50, 100 and 1,000 GB/s.
*If it fails:* fusion is only practical on fast interconnects; narrow the claim. Gate B.

**H10. Elasticity beats a static split.** A stream of GSM8K problems goes through two systems
on the same four devices, both routing with H8's entropy signal. The **swarm** fuses on demand
for escalated problems. A **static split** keeps a pipelined 6-bit model loaded at all times
and sends escalated problems to it. Splits considered: two devices for the 6-bit model plus
two solo agents, and all four devices for the 6-bit model; four solo round-to-nearest agents
with no escalation are the floor. Sweeping the escalation threshold traces an
accuracy-against-throughput curve for each system. Passes if, at H8's operating point, the
swarm's throughput **exceeds every static split that matches or beats its accuracy**. The
crossover along the sweep is reported as well. Both systems share one router, so the only
difference is elasticity. Throughput is modelled from H9. Gate C, the main result.

## Robustness

**H11. Graceful task-level degradation.** Mean GSM8K accuracy over the subsets of each size
does not decrease with group size *s* = 1 … 4, and the worst three-child group beats the best
lone child. This is the task-level version of H4.

**H12. Survival depends on churn.** With each device independently unavailable with
probability *p*, report expected accuracy and the fraction of time at least one usable answer
path exists, for the swarm, the static splits of H10, and replication at equal hardware.
Exploratory: report the crossover *p* rather than a threshold. Computed from H7 and H11's
measured accuracies.

**H13. A fused group survives a member's loss mid-generation.** In simulation, drop one member
of a four-child group halfway through each GSM8K answer. The attention cache built by the full
group is kept, and later tokens come from the three survivors. Passes a **non-inferiority
test with a 2-point margin**: the one-sided 95% confidence interval of the paired accuracy
difference (dropped mid-answer minus survivors from scratch, averaged over which member is
dropped) stays above −2 points. A plain "within 1 point" on 1,319 problems is about the size
of the noise. This is the question a real system would face: whether a cache built by one
group stays usable for a smaller one.

## Self-healing

Notation as in [method](method.md#rebuild): *Q* = Σₖ *q*ₖ is the parent's code, and every
child is a fixed function of it, *q*ₖ = ⌊(*Q* + *k*)/*N*⌋.

**H14. Exact regrowth, with storage that shrinks as the swarm grows.** Two schemes; each passes
if regrowth gives a hash match for every loss pattern it is designed for, at *N* = 4, 8 and
16, in the shard format, including codes at both ends of the range.

- *Neighbour bit (any one loss).* Given the other *N* − 1 children, a lost child's code is one
  of two adjacent values at exactly 2/*N* of the weights, and survivors can tell which weights
  those are. Child *j* + 1 stores one bit for each of them. Cost: **2/*N* bits per weight per
  child**. It cannot survive losing *j* and *j* + 1 together.
- *Coded residue (any t losses).* The residue *r* = *Q* mod *N*, together with any one
  surviving child, determines every child exactly (*q*ₖ = ⌊*Q*/*N*⌋ + [*r* + *k* ≥ *N*]).
  Spreading the residues over the children with an MDS erasure code (Reed–Solomon) lets any
  *N* − *t* survivors recover them. Cost: **log₂(*N*)/(*N* − *t*) bits per weight per child**.
  The same data also rebuilds the exact parent from any *N* − *t* children, so it subsumes the
  [coded remainder](method.md#baselines-on-the-same-nodes) baseline of phase 1.

Bits per weight per child (a 3-bit child is about 3.5):

| *N* | Neighbour bit, 1 loss | Coded residue, *t* = 1 | Coded residue, larger *t* |
|----:|------:|------:|------|
| 4 | 0.50 | 0.67 | *t* = 2: 1.00 |
| 8 | 0.25 | 0.43 | *t* = 4: 0.75 |
| 16 | 0.13 | 0.27 | *t* = 8: 0.50 |
| 100 | 0.02 | 0.07 | *t* = 25: 0.09; *t* = 50: 0.13 |

Both kinds of bits live in storage next to the shards and are never loaded for serving, so
serving memory is unchanged. For a swarm of 100, surviving the simultaneous loss of half the
members costs 0.13 bits per weight per child on disk.

**H15. No drift.** Ten rounds of losing and regrowing every member in turn, and for the coded
residue ten rounds of losing *t* random members at once, leave the parent's hash unchanged.
This follows from H14 but is tested end to end.

**H16. Fusion partners in a large swarm (exploratory).** With *N* = 16 children, compare fused
groups of four chosen at random with groups of four whose offsets are evenly spaced. The
partial-rebuild bound predicts the error depends on the largest offset gap (evenly spaced:
1/4; random four of sixteen: larger on average), so a large swarm should fuse with
complementary partners rather than any available ones. Report perplexity for each, and GSM8K
for the best and worst groups. No threshold.

**Negative control (exploratory).** Report the perplexity of one *approximate* regrowth (no
stored bits, ambiguous codes guessed), to document why the stored bits are needed. See
[known when fixing](#known-when-fixing).

## Size prediction

**H17. Fusion matters more at larger sizes.** At 27B, scaled and rotated, the worst 2-bit
child is within **1.5x** of bf16 perplexity, the same bar as H2 (9B: 3.7x), so 2-bit swarms
have usable individuals. Extrapolating the 4B → 9B trend in (ratio − 1) gives about 1.35x at
27B, so this is a real test with a modest margin. Then repeat H7 at 27B 2-bit. Phase 3.

## Order of work

1. **Gate A:** H7, H8 and H11 in simulation. H14 and H15 run alongside (code and unit tests;
   no GPU).
2. **Gate B:** H9: exact communication counts, plus the acceptance rate from simulation.
3. **Gate C:** H10 and H12, modelled from Gate A and B. H13 and H16 in simulation.
4. **Phase 3:** H17.
5. **Follow-up systems work (not this study):** real networks, containers, time to full
   health, a 100-member swarm.

If Gate A fails, the swarm framing is dropped with nothing built for it except the regrowth
code, which serves the original framing too.

## Known when fixing

**Approximate regrowth.** A Monte Carlo over 2M uniformly distributed weights, in units of the
child step and away from the ends of the code range, regrew one lost child by guessing each
ambiguous code with a fair coin. The simulation will be added as a test with the regrowth
code.

| | *N* = 4 | *N* = 8 |
|---|---:|---:|
| Weights whose lost code is ambiguous | 50% | 25% |
| Wrong after a fair guess | 25% | 12.5% |
| Regrown child, error vs original child (MSE) | 1.75x | 1.19x |
| Full rebuild with the regrown child (MSE) | 4.0x (about 1 bit of parent precision lost) | 2.5x |
| After 5 rounds of regrowing every member (rebuild MSE) | 116x | 118x |

The ambiguous codes are always one of two adjacent values, and one stored bit resolves every
one (checked exhaustively at *N* = 4 and 8). Approximate regrowth gives back a usable member
but not the fleet's precision, and repeated regrowth breaks the fleet. Self-healing therefore
has to be exact, which is why H14 is written with stored bits.

**Other results available when fixing.** The phase 1 sweeps, `seed1-9b`, `seed2-9b` and
`demo-2b` were complete; `subsets-4b`, `pool-0.8b` and `pool-2b` were still running. None
measures task accuracy.

## Decisions (2026-10-18)

- Networks are modelled rather than measured.
- Regrowth uses the neighbour bit for small swarms and the coded residue where more than one
  simultaneous loss must be survived.
- Base models, as in phase 1.
- GSM8K primary, MMLU secondary and exploratory. H10 uses a GSM8K-only stream instead of a
  mixed workload, so no "easy task" has to be chosen.

## Changes after fixing

None yet.
