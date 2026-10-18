# Pre-registration: Escalation on Disagreement (H8b, H10b)

*Proposed on 2026-10-18, after the Gate A results of the
[swarm pre-registration](preregistration-swarm.md) were seen. Fixed by the commit that adds
this file, before any of the data below exists. Nothing here is changed after a result it
governs has been seen; any later change is recorded at the end with its reason.*

## Why

H8 failed at both sizes: escalating the 30% of answers with the highest entropy recovered 36%
(9B) and 42% (4B) of the fusion gain, short of the 50% bar. As registered, H10 is therefore
dropped and stays dropped; nothing here revisits it.

An exploratory analysis of the same Gate A answers, made after H8's result was known, found a
much stronger signal: escalate when two lone agents give different final answers. With the
worst child asking and any other child as its partner, that escalated 26–27% of problems at
9B and recovered 103–104% of the gain, and 36–37% at 4B with 94–98%. It was found on the
GSM8K test split, after looking, so it is a hypothesis, not a result. This pre-registration
tests it on problems Gate A never used.

The signal needs the agents to differ: two identical replicas decoding greedily always agree.
Staggered children differ by construction, so this is a property of the swarm, not of
ensembling in general.

## Protocol

**Data.** GSM8K *train* split, problems 64 to 1,382 in dataset order (1,319 problems, as many
as the test split). The first 64 were used by the 0.8B smoke run and are skipped. Prompt,
decoding and scoring are exactly Gate A's (preregistration-swarm.md, "Changes after fixing").

**Setting.** Exactly `swarm-9b` and `swarm-4b`: 3-bit children, *g* = 64, *N* = 4, φ = ½,
rotated and scaled, the full group as the rebuilt parent.

**Conditions.** bf16 (reference only), each child alone, the full group, and a 6-bit
round-to-nearest model with the same rotation and channel scaling (the static splits' large
model).

**Agreement.** Two answers agree when both contain a final number and the numbers are equal
(as in Gate A's scoring). An answer without a number never agrees, so two answers without a
number escalate.

## H8b. Two lone agents disagreeing is a cheap escalation signal

The requester is the least accurate lone child on these problems (lowest index among ties).
For each of the other three children as its partner: the requester and partner answer alone,
and the problem goes to the full group when they disagree; otherwise the requester's answer
stands.

**Passes** if, for every partner, at most **50%** of problems are escalated and the policy
recovers at least **75%** of the gain (the full group's accuracy minus the requester's), and
that gain is at least 3 points. Escalating at random recovers about as much as the fraction
escalated, so 75% at no more than 50% clearly beats chance; random escalation is reported
alongside. The bars sit below the exploratory values to allow for regression on new data.

Evaluated separately at 9B and at 4B; each size passes or fails on its own.

## H10b. Elasticity with the disagreement router (modelled)

Four devices on H9's pass-point network (0.2 ms per step, 1 Gbit/s, 100 GB/s), with per-token
costs from H9's model at the same size (`speed-9b`, `speed-4b`):

- **Swarm.** The requester and partner answer every problem alone; on disagreement the full
  group answers. Accuracy as in H8b, averaged over the three partners.
- **Static split A.** Two devices run the requester and partner with the same router; the
  other two hold a pipelined 6-bit model that answers escalated problems.
- **Static split B.** All four devices hold a pipelined 6-bit model that answers every problem.
- **Floor.** Four lone children, no escalation (mean child accuracy).

**Throughput** is problems per second per answer token, with every device kept busy. It comes
from the device-seconds each problem needs per answer token:
- a lone answer: one solo pass on one device;
- a fused answer: four devices for one fused token (H9's fused rate with its best draft length);
- a 6-bit answer on a *d*-device pipeline: one read of the 6-bit weights plus *d* − 1 hops of one
  bf16 hidden vector.

A partition dedicated to one role uses only its own devices, so a static split is limited by
whichever partition saturates first. Answer lengths are taken as equal across systems, so they
cancel.

**Passes** (as H10 was worded) if the swarm's throughput exceeds that of every static split
whose accuracy matches or beats the swarm's. The accuracy-against-throughput points of all
four systems are reported.

## Known when fixing

**H10b is expected to fail.** An estimate from Gate A's test-split numbers and H9's model at
9B gives these per-token costs:

| | Device-ms per token | Tokens/s across 4 devices |
|---|---:|---:|
| Lone child (3.92 GB) | 39 | 102 |
| Fused group | 449 (4 devices × 112 ms) | 8.9 |
| 6-bit pipeline over 4 devices (7.28 GB) | 74 | 54 |

With 26% escalation, that comes to 20.5 problems per second per answer token for the swarm,
25.5 for split A and 54.4 for split B. Split B's accuracy is the 6-bit model's, which has not
been measured on GSM8K but is likely near bf16, as the fused group's is. Unless the 6-bit
model turns out clearly less accurate, split B beats the swarm on both axes.

The reason is structural: fusing costs about six times the device time of a 6-bit pipeline
at similar accuracy, because every device reads a whole child for every token, and the
devices exchange about 3.4 MB per token. H10b is registered anyway, so the comparison is
reported rather than chosen after the fact.

**What a reliable network leaves.** Under these assumptions the swarm's advantage over a
static split cannot be throughput. It would have to be what a pipeline lacks: answering with
any surviving device (a pipeline stops when any stage is lost), and every device being a
working agent on its own. H12 (survival under churn) measures the first and remains
exploratory in the original pre-registration.

**Disclosure.** This design, including the thresholds, was made after seeing Gate A's
results, the exploratory disagreement analysis and the estimate above.

## Order of work

1. Code: round-to-nearest conditions and a problem offset in the task harness; the H8b and
   H10b checks; configs `confirm-9b` and `confirm-4b`.
2. Run both configs on the cluster.
3. Report H8b and H10b at both sizes, whatever the outcome.

## Changes after fixing

None yet.
