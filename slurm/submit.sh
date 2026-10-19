#!/usr/bin/env bash
# Submit one run:  bash slurm/submit.sh CONFIG [sbatch options...]
# Example:         bash slurm/submit.sh configs/sweep-4b.toml --exclude=evc26
#
# Logs go to slurm_logs/<submitted YYYYmmdd-HHMM>-<config>-<job id>.out and .err, so they sort
# by submission time and name the config. The job id is also recorded in the run's runs.jsonl.
# Configs with a [live] table run on two nodes through slurm/live.sbatch; SCENARIO and NETWORK
# pick one of their runs and are added to the log name:
#                  SCENARIO=A NETWORK=lan bash slurm/submit.sh configs/live-9b.toml
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${1:?usage: bash slurm/submit.sh CONFIG [sbatch options...]}"
shift
NAME="$(basename "${CONFIG}" .toml)"
SCRIPT=slurm/sweep.sbatch
if grep -q '^\[live\]' "${CONFIG}"; then
    SCRIPT=slurm/live.sbatch
    NAME="${NAME}${SCENARIO:+-${SCENARIO}}${NETWORK:+-${NETWORK}}"
fi
LOG="slurm_logs/$(date +%Y%m%d-%H%M)-${NAME}-%j"
mkdir -p slurm_logs  # SLURM opens the log files before the job starts
# Options given after the config come last, so they override these.
sbatch --job-name="syncytium-${NAME}" --output="${LOG}.out" --error="${LOG}.err" \
    "$@" "${SCRIPT}" "${CONFIG}"
