#!/bin/bash
# Local (laptop / single-GPU box) setup: uv + optional CUDA torch + MarkLLM.
# No CARC account needed; the KGW green lists are drawn on CPU either way (see kgw_cpu.py).
#
#   bash setup_local.sh              # CUDA 12.4 wheel (RTX 30/40, A100; ~2.4 GB)
#   TORCH_INDEX=cpu bash setup_local.sh   # CPU-only wheel (~200 MB), generation is ~30x slower
#
# Then:  ollama pull qwen2.5:1.5b      (only needed for the attack step)
set -euo pipefail
cd "$(dirname "$0")"

PY=3.12
TORCH_INDEX=${TORCH_INDEX:-cu124}

if [ ! -d .venv ]; then
    uv venv --python "$PY" .venv
fi
if [ -f .venv/Scripts/python.exe ]; then VPY=.venv/Scripts/python.exe; else VPY=.venv/bin/python; fi

uv pip install --python "$VPY" torch --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
uv pip install --python "$VPY" -r requirements.txt

if [ ! -d MarkLLM ]; then
    git clone --depth 1 https://github.com/THU-BPM/MarkLLM.git MarkLLM
fi

# Qwen2.5-1.5B-Instruct: tokenizer for attacked text + a model small enough for a laptop GPU.
if [ ! -d models/Qwen2.5-1.5B-Instruct ]; then
    PYTHONIOENCODING=utf-8 "$VPY" -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='models/Qwen2.5-1.5B-Instruct',
                  allow_patterns=['*.json', '*.safetensors', '*.txt', 'merges.txt', 'vocab.json'])
"
fi

"$VPY" -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
echo
echo "verify the green lists (must print flags_md5=5d98d773dd for wm, 6375f943cb for human):"
echo "  $VPY verify_kgw.py data/wm.jsonl data/human.jsonl"
