#!/usr/bin/env bash
# One-time environment setup, run on a login node:  bash slurm/setup.sh
#
#   PYTHON       Python 3.12 interpreter used to create .venv (default: python3.12)
#   TORCH_INDEX  PyTorch wheel index matching the cluster's CUDA driver
#                (default: CUDA 12.6 wheels, which need driver 560 or newer)
set -euo pipefail

cd "$(dirname "$0")/.."
"${PYTHON:-python3.12}" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
pip install -e ".[experiments,report]"
python -c "import torch, holograph; print('torch', torch.__version__, '| holograph', holograph.__version__)"
