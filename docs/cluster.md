# Running on a SLURM cluster

The scripts in `slurm/` run one sweep config per job on a single GPU. They were written for a
cluster with 2x H100 nodes and a Lustre file system, following the conventions of other
projects on the same cluster; adjust the `#SBATCH` lines in `slurm/sweep.sbatch` for others.

## One-time setup (login node)

```bash
git clone <this repository> && cd SurviveAnything
export SBATCH_ACCOUNT=<your account>   # add to ~/.bashrc; sbatch reads it automatically
bash slurm/setup.sh                     # creates .venv and installs the package
```

If `python3.12` is not on the path, create one with Miniforge
(`conda create -n py312 python=3.12`) and pass it in:
`PYTHON=~/miniforge3/envs/py312/bin/python bash slurm/setup.sh`.

If the GPU driver is older than 560, pick an older wheel index with `TORCH_INDEX`.

## Submitting

```bash
bash slurm/submit.sh configs/sweep-4b.toml
bash slurm/submit.sh configs/sweep-9b.toml --exclude=evc33 --time=48:00:00
```

Anything after the config is passed to `sbatch`, so per-job overrides need no script edits.
Node `evc33` has been excluded by other projects on this cluster.

## Monitoring and resuming

- Logs: `slurm_logs/holograph-<config>-<job id>.out` and `.err`.
- Progress: one line per condition in the `.out` log.
- A job that hits its time limit can be resubmitted with the same command; the sweep skips
  every condition already in `results/<name>/results.jsonl`.

## Models, caches and results

- Compute nodes have internet access, so models download on first use into the Hugging Face
  cache (`~/.cache/huggingface`, on Lustre). Set `HF_HOME` to move it.
- Results are written under `results/<name>/` in the repository on Lustre. Copy them back
  for analysis, for example:
  `rsync -av <cluster>:SurviveAnything/results/sweep-9b results/`
- `holograph-report results/<name>` then writes the summary and figures to `docs/results/`.
