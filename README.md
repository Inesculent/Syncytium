# Syncytium

**Survive Anything:** low-precision LLM shards that each run on their own and together rebuild
their parent model.

> **Status: pre-alpha research.** The repository currently holds the project definition and
> tooling. No experiment code has landed yet; see the [roadmap](#roadmap).

## The idea

A trained LLM is quantized *N* times with the same step size but a different rounding offset
for each copy (*staggered quantization*). Each copy, a **child**, is an ordinary low-bit model
that runs on modest hardware. Because the offsets are staggered, the element-wise average of
all *N* children equals the parent quantized on a grid *N* times finer. The high-precision
**parent** can therefore be rebuilt by integer addition after it is lost, and losing children
lowers rebuild precision gradually instead of breaking it.

| Children | Bits per child | Rebuilt precision | Weight memory per child vs bf16 |
|---------:|---------------:|------------------:|--------------------------------:|
| 4        | 4              | ~6 bits           | ~3.5x smaller                   |
| 8        | 3              | ~6 bits           | ~4.5x smaller                   |
| 16       | 4              | ~8 bits           | ~3.5x smaller                   |

Every child holds the *whole* model at reduced resolution, and each additional child sharpens
the rebuild, which is where the name comes from.

## Documents

- [Hypothesis and goals](docs/hypothesis.md): claims, success criteria, evaluation protocol.
- [Method](docs/method.md): the quantization scheme, rebuild math and system design.
- [Plan](docs/plan.md): stages, risks and decisions.
- [Running on a cluster](docs/cluster.md): SLURM setup, submission and results.
- [Future work](docs/future-work.md): studies that can branch from this one.
- [Contributing](CONTRIBUTING.md): setup, code standards and project layout.

## Roadmap

1. **Perplexity sweep.** Qwen3.5 0.8B, 2B and 4B (Base); children at 2, 3 and 4 bits;
   *N* in {2, 4, 8}; with and without a shared rotation.
2. **Container demo.** A main container and *N* child containers. The main container is
   destroyed and its filesystem wiped; the children keep serving and rebuild it when it returns.
3. **Scale-up.** Qwen3.5 9B and larger on a GPU cluster.

## Requirements

**Hardware**

- Phase 1: an NVIDIA GPU with at least 10 GB of memory (reference machine: RTX 3080 10 GB).
  That covers the 0.8B and 2B models; the 4B model's bf16 weights alone take 9.3 GB, so 4B
  and larger run on a cluster GPU.
- Phase 2: children run on CPU through llama.cpp; the main container uses the GPU.
- Disk: about 15 GB for the 0.8B, 2B and 4B checkpoints, plus generated children.

**Software**

- Python 3.12 or newer
- PyTorch with CUDA support (phase 1)
- Docker with GPU support (phase 2): Docker Desktop with the WSL2 backend on Windows, or
  Docker Engine with the NVIDIA Container Toolkit on Linux

Python dependencies are declared in [`pyproject.toml`](pyproject.toml), split into extras so
each phase installs only what it needs:

| Extra         | Purpose                                         |
|---------------|-------------------------------------------------|
| *(core)*      | Quantization and rebuild math (PyTorch, safetensors) |
| `experiments` | Model loading and perplexity evaluation         |
| `report`      | Summary tables and figures                      |
| `export`      | Writing children as GGUF files for llama.cpp    |
| `harness`     | Container roles, heartbeats and shard transfer  |
| `dev`         | Tests, linting and pre-commit hooks             |

## Installation

```bash
python -m venv .venv                # Windows: py -3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -e ".[experiments,report,dev]"
pre-commit install
```

## Running a sweep

```bash
syncytium-sweep configs/screen-0.8b.toml   # pipeline check, about 5 minutes
syncytium-sweep configs/pilot-0.8b.toml    # full 0.8B pilot, about 2 hours on an RTX 3080
```

Each sweep writes to `results/<name>/`:

- `results.jsonl`: one row per condition (perplexity, bits per weight, offsets used).
  Conditions that produce identical weights are evaluated once and marked `reused_from`.
- `runs.jsonl`: one entry per invocation with the config, git commit, package versions and
  device.

An interrupted sweep resumes where it stopped when rerun with the same config.

```bash
syncytium-report results/pilot-0.8b        # summary table and figures in docs/results/
syncytium-compare results/pilot-0.8b results/sweep-4b results/sweep-9b   # sizes at equal memory
```

Larger models run on a SLURM cluster; see [running on a cluster](docs/cluster.md).

## Responsible scope

This project studies **fault tolerance on hardware the operator controls**. The system rebuilds
a model after hardware or container loss; it does not resist its operator. Shards move only
between explicitly enrolled nodes, nodes never start new hosts or containers themselves, and
every node obeys an operator-issued revoke command that stops it and deletes its shards.
Contributions that weaken these properties will not be accepted.

## License

[Apache License 2.0](LICENSE). A citation entry will be added with the first preprint.
