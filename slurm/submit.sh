#!/usr/bin/env bash
# Submit one run:  bash slurm/submit.sh CONFIG [sbatch options...]
# Example:         bash slurm/submit.sh configs/sweep-4b.toml --exclude=evc26
#
# Logs go to slurm_logs/<submitted YYYYmmdd-HHMM>-<config>-<job id>.out and .err, so they sort
# by submission time and name the config. The job id is also recorded in the run's runs.jsonl.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${1:?usage: bash slurm/submit.sh CONFIG [sbatch options...]}"
shift
NAME="$(basename "${CONFIG}" .toml)"
LOG="slurm_logs/$(date +%Y%m%d-%H%M)-${NAME}-%j"
mkdir -p slurm_logs  # SLURM opens the log files before the job starts
# Options given after the config come last, so they override these.
sbatch --job-name="syncytium-${NAME}" --output="${LOG}.out" --error="${LOG}.err" \
    "$@" slurm/sweep.sbatch "${CONFIG}"
