#!/usr/bin/env bash
# Submit one sweep:  bash slurm/submit.sh CONFIG [sbatch options...]
# Example:           bash slurm/submit.sh configs/sweep-4b.toml --exclude=evc33
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${1:?usage: bash slurm/submit.sh CONFIG [sbatch options...]}"
shift
mkdir -p slurm_logs  # SLURM opens the log files before the job starts
sbatch --job-name="syncytium-$(basename "${CONFIG}" .toml)" "$@" slurm/sweep.sbatch "${CONFIG}"
