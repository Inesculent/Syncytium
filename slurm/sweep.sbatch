#!/bin/bash
#SBATCH --job-name=syncytium-sweep
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
# Configs with a [pool] table are pooled-pipeline sweeps and run with syncytium-pool; configs
# with a [setting] table are swarm task runs, started as a module so they work even in an
# environment installed before their command existed.
set -euo pipefail

CONFIG="${1:?usage: sweep.sbatch CONFIG}"
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

echo "commit $(git rev-parse --short HEAD)  host $(hostname)  config ${CONFIG}"
nvidia-smi -L
# Some nodes have failed CUDA initialization (error 101); stop here so the log names the node.
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable on this node'"
if grep -q '^\[pool\]' "${CONFIG}"; then
    PYTHONUNBUFFERED=1 syncytium-pool run "${CONFIG}"
elif grep -q '^\[setting\]' "${CONFIG}"; then
    PYTHONUNBUFFERED=1 python -m syncytium.tasks.cli run "${CONFIG}"
else
    PYTHONUNBUFFERED=1 syncytium-sweep "${CONFIG}"
fi
