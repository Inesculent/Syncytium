# Pre-registration: The Swarm on Real Devices (H18)

*Proposed on 2026-10-19. Fixed by the commit that adds this file, before any live run on a 4B
or 9B model. By then the averaging layer and the packed member models had been written and
tested on a tiny CPU model only. Nothing here changes after a result it governs has
been seen; any later change is recorded at the end with its reason.*

## Why

Every swarm result so far comes from simulation: the fused group's averaged weights are
swapped into one model on one GPU. That is exact for quality, but it leaves the central claim
(that the swarm works as a distributed system that survives device loss) modelled rather than
shown. A reviewer can fairly ask whether separate devices, each holding only its own child,
really compute the parent, keep serving when members die, and heal without a backup.

H18 runs the swarm as separate processes on separate GPUs across two cluster nodes, under
scripted failures, next to the replicated-pipeline baseline of
[H12b](preregistration-survival.md) on the same hardware and the same schedule.

## Setup

**Model and children.** Qwen3.5-9B Base, 3-bit children, *g* = 64, *N* = 4, φ = ½, rotated
and channel-scaled, exactly Gate A's configuration, written as shard files with neighbour bits
and a coded residue for *t* = 2. Development uses 4B; development runs are not results.

**Devices.** Each device is one process on its own H100, with its GPU memory capped at **8 GB**
(an edge-class accelerator):
- It holds only its own child's packed codes (3.9 GB) and unpacks one layer at a time when
  computing. The bf16 parent (18 GB) cannot fit on any device.
- Four devices run on two nodes, two per node, so half the links cross the cluster network.
- Each device has its own storage directory: its shard and its regrowth data.
- A **crash** kills the process and leaves the directory intact. **Destroying** a device kills
  it and deletes the directory. A **new device** is a fresh process on a free GPU with an empty
  directory.

**Fusion.** A fused group averages the outputs of every quantized layer before the next
nonlinearity, as in the simulation. Each device sends its output to the leader (the lowest
live member), which sums the contributions in member order, divides, and returns the average.
A member that does not answer within a timeout is dropped, and the round completes with the
survivors. If the leader dies, the next lowest member takes over, and the interrupted round is
completed from contributions already sent, so every member always holds the same value.

**Control.** A controller tracks heartbeats and membership, routes requests and orchestrates
recovery:
- *Routing* (H8b): the two lowest live members answer alone; when their final numbers differ,
  every live member fuses and answers. With one member live, it answers alone.
- *Rejoining*: a device that returns with its directory is admitted only if its shard matches
  the manifest's hash.
- *Regrowth*: a new device receives a child regrown from the survivors' codes and regrowth
  data, admitted only if it matches the recorded hash.

The controller itself is not failed in these scenarios.

**Baseline.** The 6-bit round-to-nearest model (same rotation and scaling), split into two
stages and held by the same four devices: stage 1 on devices 0 and 2, stage 2 on devices 1 and
3, cross-routed through any live copy (3.64 GB per device, within the cap). It serves the same
requests under the same schedule. A lost stage copy is restored only by copying it from a
surviving copy; there is no other source.

**Networks.** Every scenario runs twice:
- on the cluster interconnect as it is;
- on an emulated wired LAN, where every message waits an extra 0.2 ms plus its size at
  1 Gbit/s (H9's pass point).

**Requests.** GSM8K test problems in dataset order, 8-shot as in Gate A, greedy, at most 256
answer tokens. A **probe** is the 8-shot prompts of problems 0–15, run teacher-forced through
the full group, recording the logits at every position. The request stream starts at problem
16.

## Scenarios

Each scenario is a sequence of phases. Every phase serves 8 requests from the stream, and the
events between phases happen at fixed points.

**Scenario A: lose two, heal with an old node and a new one.**

| Step | Event | Swarm members | Baseline |
|---|---|---|---|
| P0 | Probe, then 8 requests | 0, 1, 2, 3 | up |
| E1 | A fused request is forced; device 2 **crashes** after its 32nd answer token | 0, 1, 3 | up (stage 1 on device 0) |
| P1 | 8 requests | 0, 1, 3 | up |
| E2 | Device 0 is **destroyed** | 1, 3 | **down**: no copy of stage 1 |
| P2 | 8 requests | 1, 3 | down |
| E3 | Device 2 **returns** with its directory and is verified | 1, 2, 3 | up |
| E4 | A **new device** joins as member 0; child 0 is regrown from children 1 and 3 and the bits held by child 1 | 0, 1, 2, 3 | stage 1 copied from device 2 |
| P3 | Probe, then 8 requests | 0, 1, 2, 3 | up |

**Scenario B: no copy survives.** As A, except at E3 device 2 is **destroyed** instead of
returning, and at E4 two new devices join as members 0 and 2.
- *Swarm:* children 0 and 2 are both regrown. They are not neighbours, so neighbour bits
  suffice.
- *Baseline:* stage 1 exists nowhere, so it can never be restored, and the baseline stays down
  from E2 onward.

**Scenario C: adjacent losses.** Devices 0 and 1 are destroyed together after P0, and two new
devices join.
- *Swarm:* neighbour bits cannot regrow two adjacent children; regrowth with them must be
  refused. The coded residue (*t* = 2) regrows both.
- *Baseline:* a copy of each stage survives (devices 2 and 3), so it stays up and heals by
  copying. This scenario tests the swarm's storage choice, not the contrast.

## H18 checks

H18 passes if every check below holds in the final 9B runs of all three scenarios on both
networks. Each check is reported separately.

- **(a) Exact over the network.** On the probe, the live full group's top-1 next token agrees
  with the simulation (the same children's mean weights in one model) at **at least 99%** of
  positions. The largest and mean absolute logit differences over the last 16 positions of
  each probe prompt are reported; the full logits are too large to move. Bit identity is not
  expected: the live system averages each child's outputs, while the simulation multiplies by
  averaged weights.
- **(b) Recovery is exact.** The probe after recovery (P3, or the last probe of B and C) is
  **bit-identical** to the probe at P0, compared by the SHA-256 of every logit at every
  position: the same members, the same summation order and the same child codes must give the
  same logits.
- **(c) Healing is verified.** Every returning shard and every regrown shard matches its
  recorded SHA-256, and every refusal that the scenario calls for (neighbour bits in C) happens.
- **(d) The swarm never stops serving.** Every request in every phase gets an answer, including
  the fused request during which device 2 dies.
- **(e) The baseline behaves as its structure dictates.** It answers exactly while every stage
  has a live copy, never answers in B after E2, and in B no attempt to restore stage 1 can
  succeed.
- **(f) No device holds the parent.** Every device's peak GPU memory stays at or below 8 GB, and
  no device ever loads more than one child's codes or one stage onto its GPU. Regrowth reads
  every survivor's files in order to verify them, so a new device copies them to its disk,
  regrows, and deletes the copies; the bytes moved are reported.
- **(g) The communication model holds.** On the emulated LAN, the mean time per averaging round
  is within a factor of two of the model's prediction for the leader reduce actually used:
  two latencies plus, at the leader, (*n* − 1) incoming and (*n* − 1) outgoing vectors at the
  link's bandwidth, for *n* live members.

**Reported alongside, with no threshold:**
- a timeline figure drawn from the event log, with one lane per device, the swarm's service and
  members, and the baseline's service;
- per phase: members, requests answered, how many fused, GSM8K accuracy on those 8 requests
  (too few to compare), tokens per second solo and fused, rounds and bytes per fused token;
- recovery: time to detect a death, bytes moved, and time to regrow and verify a child;
- compute time per token, which reflects unpacking codes in Python, not low-bit kernels.

## Known when fixing

No live system existed when this was written. The quality of every member set is known from
Gate A's simulation (9B GSM8K: 86.1% with all four, 84.3–85.8% with three, 82.2–85.4% with two),
and the H9 model predicts the communication cost. This design was made after every earlier
result, including H12b, and after a back-of-envelope estimate that per-token compute will be
dominated by unpacking codes.

## Changes after fixing

**2026-10-19, before any live run on any model: how the implementation reads the design.**
These were settled while building the system and testing it on a tiny CPU model. No 4B or 9B
data existed. None of them changes a threshold.
- *The baseline runs after the swarm, not alongside it.* Each scenario is played against the
  swarm and then against the baseline, on the same four GPUs, with the same storage layout,
  requests and schedule. The baseline's E1 routes the forced request through device 2, a copy
  of stage 1, which crashes after its 32nd answer token. The pipeline then retries the request
  from the start through device 0. A request that arrives while a stage has no live copy is
  logged as unanswered at once.
- *New devices reuse freed GPUs.* The cluster's nodes have two GPUs each, so all four are in
  use. A new device starts as a fresh process with an empty directory on the GPU its
  destroyed predecessor left free.
- *Two new devices regrow independently.* In B and C each new device fetches the survivors'
  files, regrows every lost child, verifies them, and keeps only its own. The bytes moved are
  reported per device, so they are twice the minimum.
- *Choosing the regrowth data.* The controller tries neighbour bits first and the coded
  residue next. It refuses neighbour bits with the same loss check that regrowth itself
  applies, and logs each refusal; that log entry is the refusal check (c) asks for.
- *What the emulated LAN delays.* It delays every message between devices: averaging,
  pipeline hidden states and tokens, and file transfers. It does not delay messages to and
  from the controller. A probe's averaged logits take about ten seconds to cross a 1 Gbit/s
  link, and a heartbeat queued behind them would get a live device taken for dead. A
  controller is not one of the devices whose network is being emulated. Every timeout between
  devices also allows the time the emulated link needs to move that round's vectors through
  the leader. Each device runs one token alone after loading, so start-up costs never fall
  inside a timed or fused pass.
- *Detection.* Devices send a heartbeat every 0.5 s, and one silent for 2 s is taken for dead.
  Detection time runs from the kill command. For E1's victim, which crashes on its own, it runs
  from the victim's last heartbeat. If the victim's answer ends before 32 tokens, the victim is
  crashed right after and the log says so.
- *Check (a).* The simulation's logits come from the parent's bf16 weights (the mean of the
  four children) in the bf16 model, teacher-forced on the same token ids. Only P0's probe keeps
  the last 16 positions' logits; later probes keep top-1 tokens and hashes, which is all (b)
  needs.
- *Check (f).* Peak memory is PyTorch's peak reserved memory per device process, which the cap
  bounds; the CUDA context is not counted. A device process builds exactly one child or stage
  from its own file, so "no more than one" holds by construction; the largest packed size
  loaded is reported.
- *Check (g).* The averaging rounds of the fused requests count, not the probes'. Each
  request's first round is left out, because it also waits for every member to start. The
  measured total time is compared with the model's prediction for the same rounds, each with
  its own vector size and number of live members.
- *Pipeline decoding.* Stage 1 drives greedy decoding. Each stage keeps its own cache, and
  token positions are passed explicitly. On the tiny model, two stages give the whole 6-bit
  model's logits and answers.

**2026-10-21, after the first 4B development run: how a group treats a slow member.** The
first 4B run of scenario A on the cluster network stopped during P0's probe, before any
request was served. Development runs are not results, and nothing here changes a threshold.
- *What happened.* Every averaging round of a request was numbered from 1 again, and a leader
  that gave up on a member did not tell it. A member still alive kept waiting, took the leader
  for dead and tried to lead itself, and a contribution it sent late was taken, in a later
  probe prompt, for that prompt's own. Every device then failed.
- *Messages belong to one request.* Every averaging message carries its request's tag, and
  messages from earlier requests are discarded.
- *A dropped member leaves.* A member the leader drops while it is still alive receives the
  round's result without itself among the members. It stops taking part in that request and
  answers the controller for itself only, and the group's answer comes from the rest. Such
  drops are logged.
- *Waits allow for the real network.* A probe averages the logits at every position, about
  0.4 GB per member per prompt for the 4B model. On top of the emulated link, every wait now
  allows for the real network moving the round's vectors through the leader at 1 Gbit/s or
  faster. Before, it allowed nothing on the cluster network.
- *Checks (a) and (b) need the full group.* A probe reports whether every prompt finished with
  every member it started with, and (a) and (b) fail if it did not.
