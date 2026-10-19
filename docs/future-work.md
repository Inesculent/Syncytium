# Future Work

Studies that can branch from the core result, grouped by theme. Each states the question and
why it matters.

## The swarm

- **Real devices and networks.** Fused agents on separate machines over a LAN and over
  Wi-Fi: tokens per second, bytes per token and time to recover a member, against the
  speed model of H9.
- **Instruction-tuned agents in an agent loop.** Multi-step tasks where agents work alone,
  fuse when they disagree, and keep going while members fail and regrow.
- **Cheaper fusion.** Fusing costs about six times the device time of a pipeline at similar
  accuracy. Averaging only some layers, exchanging fewer logits, or fusing two members
  instead of all could narrow that gap.
- **Routing beyond disagreement.** Disagreement is confirmed (H8b). Open: how many agents
  should answer before deciding, and whether partial fusion (two or three members) is enough
  for most escalations.
- **Large swarms.** Which members to fuse when only some are reachable (H16), how the
  coded residue scales to hundreds of members, and placement across failure domains.
- **Channel scales in shards.** Scaled children are the best ones, but shard files cannot
  store the shared scales yet, so the console's swarms are unscaled.

## Precision and quality

- **Bit-exact recovery.** Store the bits below the rebuilt grid as an erasure-coded residual
  spread across children, so the original bf16 weights come back exactly at small extra cost.
- **Better offset designs.** Uniform staggering is optimal for scalar rounding, but lattice and
  vector quantizers from multiple description coding may do better per bit, especially at 2
  bits.
- **Heterogeneous children.** Give larger devices more bits and smaller ones fewer, and work
  out how mixed bit widths combine in a rebuild.
- **Quantization-aware training for staggered children.** Fine-tune so that every offset grid
  rounds well at once, improving children without breaking the rebuild identity.
- **Staggered KV-cache and activation quantization.** Extend the scheme beyond weights to cut
  runtime memory on small devices.

## Model structure

- **Width split combined with precision split.** Divide attention heads and FFN neurons across
  children (trained with structured dropout) and stagger precision as well, giving a
  two-dimensional trade-off between width and bits.
- **Mixture-of-experts models.** Distribute experts and staggered copies across nodes, and
  study whether expert routing tolerates missing or low-precision experts.
- **Scaling behavior.** How child quality and rebuild gains change from 1B to 30B+ parameters.

## Learning and merging

- **Continual learning on children.** Let children fine-tune locally (for example with LoRA)
  and merge their updates into the parent. Key questions: consistency of shared quantization
  parameters, versioned manifests, and how merging methods (FedAvg, TIES, task arithmetic)
  interact with staggered grids.
- **Regenerating lost information.** When every copy of some information is gone, regrow it by
  distilling from the surviving model and data rather than recovering it exactly.
- **Output ensembling.** Compare averaging children's *outputs* (logits) at inference against
  rebuilding the parent from their *weights*, in quality and cost.

## Systems

- **Heterogeneous home networks.** Phones, laptops, single-board computers and NAS devices:
  rebuild time under realistic bandwidth, churn and device loss.
- **Progressive rebuild.** Serve from the best available partial rebuild and refine it as
  children reconnect, measuring quality over time after a failure.
- **Distributed inference without a rebuild.** Whether children can cooperate on a single
  forward pass efficiently, compared with tensor and expert parallelism. Fused agents do
  this in simulation; H9's model says it is affordable on a LAN for 9B but not efficient.

## Security and governance

- **Shard confidentiality.** Apply an all-or-nothing transform so that fewer than *k* shards
  reveal nothing about the weights, while children remain runnable.
- **Integrity.** Signed manifests and detection of a tampered or poisoned child before it
  contaminates a rebuild.
- **Operator control under failure.** Formal guarantees that revoke and shutdown commands
  reach every node and take effect even during partitions, rebuilds and regrowth: a swarm
  that regrows members must never regrow one the operator has revoked.
