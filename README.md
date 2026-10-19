# Syncytium

**Survive Anything:** a swarm of low-bit LLM agents that each run on their own, fuse into
their full-precision parent when a task needs it, and regrow any member that is lost.

> **Status: pre-alpha research, studied in simulation.** Start with the
> [research report](docs/report.md): what was done, why, what came of it and what it means, in
> one place. In short, fusion matches the parent on GSM8K at 4B and 9B, the swarm survives
> losing members, lost members are regrown exactly, and two agents disagreeing tells the
> swarm when to fuse. Fusing is slower than a pipeline, but under 10–30% device loss with no
> backup the swarm answers more accurately than replicated pipelines (H12b). 2-bit children
> are still not usable alone at 27B (H17). See the [roadmap](#roadmap).

## The idea

A trained LLM is quantized *N* times with the same step size but a different rounding offset
for each copy (*staggered quantization*). Each copy, a **child**, is an ordinary low-bit model
that runs on a small device. Because the offsets are staggered, the children's integer codes
add up exactly to the parent quantized on a grid *N* times finer, and the average of their
weights is that parent.

That makes a set of children a **syncytium**: many agents that are also one.

- **Alone**, each child is a working agent: a 3-bit 9B child solves 77–82% of GSM8K
  (bf16: 86%).
- **Fused**, any group of live children averages its layer outputs and runs as one more
  precise model, with no extra memory and no weights to load. Four fused 3-bit 9B children
  solve 86.1%.
- **Losing members** lowers precision step by step instead of breaking anything. A fused group
  that loses a member mid-answer finishes the answer without loss.
- **Regrowing members** is exact: survivors rebuild a lost child bit for bit, checked by hash,
  from a small amount of stored data (0.5 bits per weight per child for four children, less
  for larger swarms).

The goal is **survival, not efficiency**. Fusing is slower than running a pipeline over the
same devices. What it buys is a model that keeps answering with whatever survives, gets
better with every member present, and heals.

| Children | Bits per child | Fused precision | Weight memory per child vs bf16 |
|---------:|---------------:|----------------:|--------------------------------:|
| 4        | 3              | ~5 bits         | ~4.5x smaller                   |
| 4        | 4              | ~6 bits         | ~3.5x smaller                   |
| 16       | 4              | ~8 bits         | ~3.5x smaller                   |

## Documents

- [Research report](docs/report.md): the whole project in one place, with conclusions,
  limitations and the status of every hypothesis.
- [Swarm report](docs/results/swarm-report.md): fusion, survival and regrowth at 4B and 9B,
  with every pre-registered outcome.
- [Phase 1 report](docs/results/phase-1-report.md): the perplexity sweeps behind the scheme.
- Pre-registrations, fixed before their data existed:
  - [Hypothesis and goals](docs/hypothesis.md): H1–H6;
  - [swarm](docs/preregistration-swarm.md): H7–H17;
  - [escalation on disagreement](docs/preregistration-disagreement.md): H8b, H10b;
  - [survival under device loss](docs/preregistration-survival.md): H12b.
- [Method](docs/method.md): quantization, rebuild, fusion, regrowth and the speed model.
- [Plan](docs/plan.md): stages, risks and decisions.
- [Running on a cluster](docs/cluster.md): SLURM setup, submission and results.
- [Future work](docs/future-work.md): studies that can branch from this one.
- [Contributing](CONTRIBUTING.md): setup, code standards and project layout.

## Roadmap

1. **Perplexity sweeps.** Qwen3.5 0.8B, 2B, 4B and 9B (Base); 2–4-bit children; *N* in
   {2, 4, 8}; shared rotation and activation-aware channel scaling; pooled pipelines. Done.
2. **The swarm, in simulation.**
   - Done: task accuracy of lone and fused agents, members lost mid-answer, the speed of
     fused agents under a network model, and exact regrowth.
   - Done: escalation on two agents' disagreement, confirmed on fresh problems.
   - Done: survival under device loss against replicated pipelines (H12b holds).
   - Next: correlated failures, and choosing fusion partners in larger swarms (H16).
3. **Scale and systems.** Done: the post-trained series up to 27B (H17 fails: 2-bit children
   are not usable alone at 27B). Next: a 3-bit swarm at 27B, real networks and devices,
   instruction-tuned agents.

## Requirements

**Hardware**

- An NVIDIA GPU for experiments. A 10 GB card covers 0.8B and 2B; 4B and larger run on a
  cluster GPU.
- The swarm console runs on CPU for small models.
- Disk: about 15 GB for the 0.8B, 2B and 4B checkpoints, plus generated children.

**Software**

- Python 3.12 or newer
- PyTorch with CUDA support for experiments

Python dependencies are declared in [`pyproject.toml`](pyproject.toml), split into extras so
each use installs only what it needs:

| Extra         | Purpose                                         |
|---------------|-------------------------------------------------|
| *(core)*      | Quantization, rebuild and regrowth math (PyTorch, safetensors) |
| `experiments` | Model loading, perplexity and task evaluation   |
| `report`      | Summary tables and figures                      |
| `export`      | Writing children as GGUF files for llama.cpp    |
| `harness`     | Container roles, heartbeats and shard transfer (deferred) |
| `dev`         | Tests, linting and pre-commit hooks             |

## Installation

```bash
python -m venv .venv                # Windows: py -3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -e ".[experiments,report,dev]"
pre-commit install
```

## Querying a swarm

```bash
syncytium-swarm create Qwen/Qwen3.5-0.8B-Base swarms/0.8b   # 3-bit children + regrowth data
syncytium-swarm console swarms/0.8b                          # type help
```

In the console:
- `ask` lets the leader answer, and fuses every live member if it is unsure;
- `solo K` and `fuse all|K,K,…` choose who answers;
- `kill K` loses a member (its files move to `lost/`);
- `regrow` rebuilds lost members from the survivors, checked against their hashes.

Weights are swapped into one model, so the console shows what the swarm answers, not how fast
it would be on separate devices.

## Running experiments

Every experiment is a TOML config in `configs/`, writes to `results/<name>/`, and resumes where
it stopped when rerun.

```bash
syncytium-sweep configs/pilot-0.8b.toml           # perplexity of children and rebuilds
syncytium-pool run configs/pool-0.8b.toml         # pooled pipeline with lost devices
syncytium-tasks run configs/swarm-9b.toml         # GSM8K and MMLU of lone and fused agents
syncytium-speed run configs/speed-9b.toml         # drafting agreement for the speed model
```

Reports go to `docs/results/<name>/`:

```bash
syncytium-report results/pilot-0.8b
syncytium-pool report results/pool-0.8b
syncytium-tasks report results/swarm-9b
syncytium-speed report configs/speed-9b.toml
syncytium-speed confirm configs/speed-9b.toml results/confirm-9b    # H8b, H10b
syncytium-speed survival configs/speed-9b.toml results/swarm-9b     # H12b
```

Each run also writes `runs.jsonl`, with the config, git commit, package versions, device and
SLURM job of every invocation. `syncytium-runs` lists every config with how far its run got
locally, so it shows what is finished, what is still on the cluster, and what is unreported. Larger models run on a SLURM cluster; see
[running on a cluster](docs/cluster.md).

## Responsible scope

This project studies **fault tolerance on hardware the operator controls**. The swarm survives
hardware and device loss; it does not resist its operator. Shards move only between explicitly
enrolled nodes, nodes never start new hosts or containers themselves, and regrowth only
rebuilds members the operator's roster already names. Every node obeys an operator-issued
revoke command that stops it and deletes its shards, and a revoke must win over any regrowth
in progress. Contributions that weaken these properties will not be accepted.

## License

[Apache License 2.0](LICENSE). A citation entry will be added with the first preprint.
