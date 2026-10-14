#!/bin/bash
# One-time setup of the shared group4 Python env on CARC (run on the login node).
# torch 2.5.1 + cu121 still ships kernels for P100 (sm_60) and V100 (sm_70), so jobs can use any GPU
# in the gpu partition; torch >= 2.8 (cu128) only runs on A40/A100/L40S.
#   bash setup_env.sh
set -euo pipefail
umask 002
ENV=${ENV:-/project2/xiangren_1987/group4/envs/wm}

module load python/3.11.9
python -m venv "$ENV"
source "$ENV/bin/activate"
pip install --no-cache-dir -q --upgrade pip
pip install --no-cache-dir -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir -q transformers==4.46.3 accelerate sentence-transformers==3.3.1 nltk datasets
python -c "import torch, transformers; print('torch', torch.__version__, 'arch', torch.cuda.get_arch_list(), 'transformers', transformers.__version__)"
