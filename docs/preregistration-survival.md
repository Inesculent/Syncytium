# Pre-registration: Survival under Device Loss (H12b)

*Proposed on 2026-10-18. Fixed by the commit that adds this file, before the 6-bit model's
GSM8K accuracy, the one measurement H12b still needs, exists. Nothing here changes after a
result it governs has been seen; any later change is recorded at the end with its reason.*

## Why

The [swarm report](results/swarm-report.md) concludes that the swarm is survivable, not
efficient. On a reliable network, a 6-bit model pipelined over the same devices is several
times faster at similar accuracy (H10b is registered as expected to fail). The swarm's case
therefore rests on what happens when devices are lost. H12 in the
[swarm pre-registration](preregistration-swarm.md) measured this only as an exploratory
comparison with no threshold. H12b makes it confirmatory, against the strongest alternative.

## Scenario

Four devices, each with the memory of one 3-bit child, and **no copy of the weights anywhere
else**. During a window, each device is lost independently with probability *p*, and nothing
lost is restored until the window ends. A request during the window is answered by whatever
is still running.

- **Swarm.** Each device holds one staggered 3-bit child (*N* = 4, φ = ½, rotated and scaled, as
  in Gate A). Every request is answered by the fused group of all live members. If no member
  is live, the request goes unanswered.
- **Replicated pipelines** (the baseline the swarm must beat). The 6-bit round-to-nearest model,
  rotated and scaled with the children's settings, is split into two stages. Each stage is held
  by two devices: stage 1 on devices 0 and 2, stage 2 on devices 1 and 3. A request is answered
  when at least one copy of each stage is live, routed through any live copy (as Petals does).
  Each half fits one device: 3.64 GB against a 3.92 GB child at 9B, and 1.71 GB against
  1.84 GB at 4B.

## Metric

**Expected per-request accuracy**: the probability that a request is answered, times the
accuracy of the answer, with unanswered requests counted as wrong. It is computed exactly over
all 16 states of the four devices:
- each state is weighted by *p*^(lost) (1 − *p*)^(live);
- for the swarm, each state uses the measured GSM8K accuracy of exactly the live members fused
  together, from Gate A, where every subset was measured;
- for the pipelines, each state uses the 6-bit model's accuracy, or zero when a stage has no
  live copy.

**Data.** GSM8K test accuracies at 9B and 4B from `swarm-9b` and `swarm-4b`, plus the 6-bit
round-to-nearest model's accuracy on the same 1,319 problems. That one condition per size is
added to those configs (`rtn_bits = [6]`); the runner resumes and evaluates only it.

## H12b

**Passes** at a model size if the swarm's expected per-request accuracy exceeds the replicated
pipelines' at every *p* in {0.1, 0.2, 0.3}. Each size passes or fails on its own and is
reported on its own: the swarm is meant for larger models and larger swarms than either
size here, so neither size stands for the other.

The pass range starts at 10% because that is the failure rate of the devices a swarm is for:
phones, laptops, edge boxes and volunteer machines are often unavailable more than a tenth of
the time, while datacenter servers, where a pipeline belongs, are far below 1%. This was
chosen after seeing the estimate below, which puts the 9B crossover near 5%.

**Reported alongside, with no threshold:**
- the full curves for *p* from 0.001 to 0.3, and the crossover *p*;
- the probability of answering and the accuracy when answered, separately;
- **throughput**: correct answers per second per answer token, with H10b's costs. The swarm is
  expected to lose this at every *p*;
- **restoring full health with no backup**, the probability that everything lost can be rebuilt
  from what survives:
  - swarm with neighbour bits: no two cyclically adjacent members lost;
  - swarm with the coded residue: at most *t* members lost, for *t* = 1, 2, 3;
  - pipelines: each stage keeps a live copy.

  It also reports how well the swarm's survivors still answer when full health cannot be
  restored.

## Known when drafting

**An estimate puts the 9B crossover at the bottom of the pass range.** It is computed exactly
over the 16 states from Gate A's measured accuracies, but *assumes* the 6-bit model scores
like bf16 (9B 86.2%, 4B 82.6%); that accuracy has not been measured.

| *p* | 0.01 | 0.03 | 0.05 | 0.1 | 0.2 | 0.3 |
|---|---:|---:|---:|---:|---:|---:|
| 9B swarm | 0.860 | 0.859 | **0.858** | 0.856 | 0.850 | 0.839 |
| 9B replicated pipelines | 0.862 | 0.861 | **0.858** | 0.845 | 0.794 | 0.714 |
| 4B swarm | 0.828 | 0.827 | 0.826 | 0.822 | 0.813 | 0.797 |
| 4B replicated pipelines | 0.826 | 0.824 | 0.822 | 0.809 | 0.761 | 0.684 |

- The pipelines answer with probability (1 − *p*²)². The swarm answers unless all four devices
  are lost, but its accuracy drops as members go.
- At 9B, *p* = 0.05 separates them by 0.0006 (0.06 points), so the outcome there hinges on the
  unmeasured 6-bit accuracy. H12b is a real test.
- At 4B the swarm leads throughout, because its fused group (82.9%) slightly beats bf16
  (82.6%), a difference within sampling noise.

**The two choices that decide the outcome:**
- *Cross-routed replicas.* If each replica pipeline could only use its own pair of devices, they
  would answer with probability 1 − (1 − (1 − *p*)²)², and the 9B crossover would fall to about
  *p* = 0.03. Cross-routing is the stronger baseline, so it is the one registered.
- *The fused group.* The swarm always fuses every live member, which gives the best answer per
  request. Its throughput cost is reported, not hidden.

**The 6-bit model's accuracy on other problems was seen before committing.** The pass range
and every other choice here were settled before any 6-bit GSM8K number existed. Before this
file was committed, though, the `confirm-*` runs of the
[disagreement pre-registration](preregistration-disagreement.md) measured the 6-bit model on
GSM8K *train* problems 64–1,382. It matched bf16 within 0.1 point: 87.95% against 87.87% at
9B, 84.15% against 84.23% at 4B. H12b uses the *test* problems, where the 6-bit model has
still not been run. But if it matches bf16 there too, the estimate above already says H12b
passes at both sizes. Nothing was changed after seeing these numbers.

**Disclosure.** This design, including the 10% bottom of the pass range, was chosen after
seeing Gate A's results and the estimate above. By that estimate, H12b passes at both sizes
unless the 6-bit model is more accurate than bf16 by more than 1.2 points at 9B or 1.3 points
at 4B; p = 0.1 is the binding rate.

## Order of work

1. Fix this file by committing it.
2. Add `rtn_bits = [6]` to `configs/swarm-9b.toml` and `configs/swarm-4b.toml`, and resubmit
   them; only the 6-bit condition runs.
3. Code the exact state enumeration and the report (no GPU needed).
4. Report H12b at both sizes, whatever the outcome.

## Changes after fixing

None yet.
