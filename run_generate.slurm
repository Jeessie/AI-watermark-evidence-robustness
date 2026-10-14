#!/bin/bash
# A2 on USC CARC (Discovery). Submit from the project folder:
#   sbatch run_generate.slurm                 # all 200 prompts
#   sbatch run_generate.slurm --limit 10      # pilot
# Extra arguments are passed through to generate.py. Resubmitting resumes from data/wm.jsonl.
#SBATCH --job-name=kgw-gen
#SBATCH --account=CHANGE_ME          # your CARC project account, see `myaccount`
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
mkdir -p logs

# Keep the model download out of the small home quota.
export HF_HOME=${HF_HOME:-/scratch1/$USER/hf_cache}

source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
python generate.py --markllm_dir MarkLLM "$@"
