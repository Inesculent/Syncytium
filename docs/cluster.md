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

Pooled-pipeline configs go through the same script: a config with a `[pool]` table runs
with `syncytium-pool run`, every other config with `syncytium-sweep`.

```bash
bash slurm/submit.sh configs/pool-0.8b.toml
```

Before evaluating anything, the job checks that PyTorch sees a GPU. Some nodes have failed
CUDA initialization (error 101); such a job now stops at once with `CUDA unavailable on this
node` in its `.err` log, and the node can be excluded with `--exclude=<node>` on
resubmission.

## Monitoring and resuming

- Logs: `slurm_logs/syncytium-<config>-<job id>.out` and `.err`.
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
  for pooled sweeps, `syncytium-pool report results/<name>`.
