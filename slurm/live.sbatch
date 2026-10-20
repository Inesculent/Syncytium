#!/bin/bash
#SBATCH --job-name=syncytium-live
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#
# Runs H18's live scenarios (docs/preregistration-demo.md) on two nodes with two GPUs each.
# Submit through slurm/submit.sh, which sends configs with a [live] table here. SCENARIO
# (A, B or C) and NETWORK (cluster or lan) pick one run; without them the job runs them all.
# The shards and the simulation reference are written once, by the first job, and reused.
set -euo pipefail

CONFIG="${1:?usage: live.sbatch CONFIG}"
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "commit $(git rev-parse --short HEAD)  nodes ${SLURM_JOB_NODELIST}  config ${CONFIG}"
echo "scenario ${SCENARIO:-all}  network ${NETWORK:-all}"
# Steps take no CPU binding: on some nodes the job's binding does not fit a step that holds
# part of its GPUs ("Unable to satisfy cpu bind request").
ONE_GPU=(srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpu-bind=none)
"${ONE_GPU[@]}" nvidia-smi -L
# Some nodes have failed CUDA initialization (error 101); stop here so the log names the node.
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:2 --cpu-bind=none \
    python -c "import torch; assert torch.cuda.device_count() == 2, 'CUDA unavailable on this node'"

"${ONE_GPU[@]}" python -m syncytium.live.run prepare "${CONFIG}"
"${ONE_GPU[@]}" python -m syncytium.live.run reference "${CONFIG}"

ARGS=()
if [[ -n "${SCENARIO:-}" ]]; then ARGS+=(--scenario "${SCENARIO}"); fi
if [[ -n "${NETWORK:-}" ]]; then ARGS+=(--network "${NETWORK}"); fi
python -m syncytium.live.run run "${CONFIG}" "${ARGS[@]}"
python -m syncytium.live.run check "${CONFIG}"
