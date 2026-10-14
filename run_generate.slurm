#!/bin/bash
# A2 on USC CARC (Discovery). Submit from /project2/xiangren_1987/group4/AI-watermark-evidence-robustness:
#   sbatch run_generate.slurm                 # all 200 prompts
#   sbatch run_generate.slurm --limit 10      # pilot
#   PY=/path/to/python sbatch run_generate.slurm ...   # use another env (needs torch <= 2.7 to run on P100/V100)
# Extra arguments are passed through to generate.py. Resubmitting resumes from data/wm.jsonl.
#SBATCH --job-name=kgw-gen
#SBATCH --account=xiangren_1987      # CSCI 544 course account (not your lab default)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
umask 002   # keep outputs writable by the rest of group4

# Models are pre-downloaded into the group cache (see README), so run offline.
export HF_HOME=${HF_HOME:-/project2/xiangren_1987/group4/hf_cache}
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Shared group env from setup_env.sh (torch 2.5.1 + cu121 runs on any GPU in the partition).
PY=${PY:-/project2/xiangren_1987/group4/envs/wm/bin/python}
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY generate.py --markllm_dir MarkLLM "$@"
