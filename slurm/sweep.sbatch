#!/bin/bash
#SBATCH --job-name=holograph-sweep
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#
# Runs one sweep config on one GPU. Submit through slurm/submit.sh, which creates the log
# directory first. The SLURM account is read from SBATCH_ACCOUNT (see docs/cluster.md).
set -euo pipefail

CONFIG="${1:?usage: sweep.sbatch CONFIG}"
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

echo "commit $(git rev-parse --short HEAD)  host $(hostname)  config ${CONFIG}"
nvidia-smi -L
PYTHONUNBUFFERED=1 holograph-sweep "${CONFIG}"
