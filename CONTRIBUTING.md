# Contributing

## Setup

Follow the installation steps in the [README](README.md), then install the git hooks:

```bash
pre-commit install
```

Before opening a pull request, run:

```bash
ruff check . && ruff format --check . && pytest
```

## Code standards

The code accompanies a paper, so readers should be able to check it against the
[method](docs/method.md) line by line.

- **Small units.** One responsibility per module and per function. Ruff enforces limits on
  complexity, arguments, branches and statements; split code rather than suppress the check.
- **Types and docstrings.** Public functions have type hints and a Google-style docstring that
  names the symbols from the method document (Δ, *N*, *b*, *q*ₖ) where relevant.
- **Math stays pure.** Quantization and rebuild functions take tensors and return tensors. File
  I/O, model loading and networking live in separate modules.
- **Configuration over constants.** Experiment settings come from config files in `configs/`,
  not from values edited inside scripts.
- **Tests for every formula.** Each piece of quantization math has a unit test, including
  edge cases (constant groups, extreme values, *N* = 1).
- **No notebooks in the package.** Exploratory work stays out of `src/`. If a notebook is
  committed, its outputs are stripped.
- **No weights or raw results in git.** Models, data and run outputs are gitignored; summaries
  and figures that support the paper go in `docs/`.

## Experiments

- Every run is launched from a config file and writes its resolved config, seed, package
  versions and git commit next to its results.
- Results that appear in the paper must be reproducible from a tagged commit and a config.

## Project layout

Current:

```
configs/                sweep configs (TOML)
docs/                   hypothesis, method, plan and future work
src/holograph/quant/    staggered quantization, rebuild, partial rebuild, simulation
src/holograph/models/   loading the text stack, choosing weights, swapping weights in place
src/holograph/eval/     corpus, perplexity and memory accounting
src/holograph/sweep/    sweep configs, conditions, results log, runner and CLI
tests/                  unit tests, mirroring the package layout
```

Planned as work lands:

```
src/holograph/export/   GGUF writer for children
src/holograph/harness/  container roles, heartbeats, shard transfer, operator controls
docker/                 Dockerfiles and compose files
```

## Commits

Use short, imperative subject lines ("Add partial rebuild decoder"). Keep each commit to one
logical change.

## Scope

Contributions must keep the guarantees described in the README's
[responsible scope](README.md#responsible-scope) section.
