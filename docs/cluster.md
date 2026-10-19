# Running on a SLURM cluster

The scripts in `slurm/` run one sweep config per job on a single GPU. They were written for a
cluster with 2x H100 nodes and a Lustre file system, following the conventions of other
projects on the same cluster; adjust the `#SBATCH` lines in `slurm/sweep.sbatch` for others.

## One-time setup

```bash
git clone <this repository> && cd Syncytium
export SBATCH_ACCOUNT=<your account>   # add to ~/.bashrc; sbatch and srun read it automatically
srun --qos=normal --cpus-per-task=6 --mem=32G --time=01:00:00 --pty bash
bash slurm/setup.sh                     # creates .venv and installs the package
```

Run the setup inside the interactive allocation, not on the login node. Login nodes cap
virtual memory, so the script's final `import torch` fails there with `libtorch_cuda.so:
failed to map segment from shared object` even though the install itself succeeded. The
same applies to anything else that imports torch, such as `pytest` or `syncytium-report`.

If `python3.12` is not on the path, create one with Miniforge
(`conda create -n py312 python=3.12`) and pass it in:
`PYTHON=~/miniforge3/envs/py312/bin/python bash slurm/setup.sh`.

If the GPU driver is older than 560, pick an older wheel index with `TORCH_INDEX`.

## Submitting

```bash
bash slurm/submit.sh configs/sweep-4b.toml
bash slurm/submit.sh configs/sweep-9b.toml --exclude=evc26 --time=48:00:00
```

Anything after the config is passed to `sbatch`, so per-job overrides need no script edits.
Exclude node `evc26`, which has been failing jobs. Node `evc33`, which other projects on this
cluster used to exclude, was fixed by 2026-10-18.

Every kind of config goes through the same script, which picks the command from the
config's tables:

| Table in the config | Runs |
|---|---|
| `[pool]` | `syncytium-pool run` (pooled pipeline) |
| `[acceptance]` | `python -m syncytium.speed.cli run` (drafting agreement for H9) |
| `[setting]` | `python -m syncytium.tasks.cli run` (GSM8K and MMLU of lone and fused agents) |
| anything else | `syncytium-sweep` |

```bash
bash slurm/submit.sh configs/pool-0.8b.toml
bash slurm/submit.sh configs/swarm-9b.toml --exclude=evc26
```

Task and speed runs start as modules because an editable install picks up new code but
not new commands: a command added after `slurm/setup.sh` ran does not exist until the
package is reinstalled with `pip install -e ".[experiments,report]"`. That is how the
first pool runs failed with `syncytium-pool: command not found`.

Before evaluating anything, the job checks that PyTorch sees a GPU. Some nodes have failed
CUDA initialization (error 101); such a job now stops at once with `CUDA unavailable on this
node` in its `.err` log, and the node can be excluded with `--exclude=<node>` on
resubmission.

## Run names and logs

A run is named after its config, `<study>-<size>`, and writes to `results/<name>/`. The study
says what the run tests:

| Study | What it runs | Hypotheses |
|---|---|---|
| `pilot`, `sweep`, `screen` | Perplexity of children and rebuilds | H1–H4 |
| `subsets` | Every survivor count | H4, placement rule |
| `scaling`, `seed1`, `seed2` | Channel scaling; rotation seeds | Stage 1.9 |
| `demo` | The container demo's scheme | — |
| `pool`, `pool-screen` | Pooled pipeline | H6 |
| `swarm` | Gate A: lone and fused agents on GSM8K and MMLU | H7, H8, H11, H13, H12b |
| `speed` | Drafting agreement for the speed model | H9 |
| `confirm` | Escalation on disagreement, fresh problems | H8b, H10b |
| `post` | Post-trained series | H17 |
| `live` | The swarm on real devices, against the replicated pipeline | H18 |

A new config takes an existing study prefix, or a new one added to this table, and opens
with a comment whose first sentence names its hypothesis and what it measures.

- **Status.** `syncytium-runs` lists every config with its kind, how many of its conditions
  are logged locally, the commit and SLURM job of its latest invocation, and whether it is
  reported. `--pending` keeps only runs not complete here. Results directories that no
  config writes to are listed as strays. Run it after copying results back to see what is
  still on the cluster.
- **Logs.** `slurm/submit.sh` writes
  `slurm_logs/<submitted YYYYmmdd-HHMM>-<config>-<job id>.out` and `.err`, and every
  invocation records its job id and node in `results/<name>/runs.jsonl`. So a results
  directory leads straight to its log.

## Live runs (H18)

A config with a `[live]` table runs on **two nodes with two GPUs each** through
`slurm/live.sbatch`. It plays one scenario on one network against the swarm and then the
replicated pipeline, so submit one job per scenario and network:

```bash
SCENARIO=A NETWORK=cluster bash slurm/submit.sh configs/live-4b.toml --exclude=evc26
```

The log name gains the scenario and network (`...-live-4b-A-cluster-<job id>.out`). The first
job writes the shard sets under `shards/live-<size>/` and the simulation reference, which takes
one GPU for a while; submit the other jobs once its log shows the scenario has started, so two
jobs never write the same shards. Each job ends by recomputing `results/<name>/checks.json`
and the timeline figures from every run so far.

Inside the job, the runner starts one host per node (`python -m syncytium.live.host`), and the
hosts start each device as its own process on one GPU, with node-local storage under
`/tmp/syncytium-live`. Every device's output goes to
`results/<name>/<scenario>-<network>/<system>/devices/`.

## Monitoring and resuming

- Logs: `slurm_logs/<submitted YYYYmmdd-HHMM>-<config>-<job id>.out` and `.err`.
- Progress: one line per condition in the `.out` log.
- A job that hits its time limit can be resubmitted with the same command; the sweep skips
  every condition already in `results/<name>/results.jsonl`.

## Models, caches and results

- Compute nodes have internet access, so models download on first use into the Hugging Face
  cache (`~/.cache/huggingface`, on Lustre). Set `HF_HOME` to move it.
- Results are written under `results/<name>/` in the repository on Lustre. Copy them back
  for analysis, for example:
  `rsync -av <cluster>:Syncytium/results/sweep-9b results/`
- `syncytium-report results/<name>` then writes the summary and figures to `docs/results/`;
  for pooled sweeps, `syncytium-pool report results/<name>`; for task runs,
  `syncytium-tasks report results/<name>`; for speed runs,
  `syncytium-speed report configs/<name>.toml`.
