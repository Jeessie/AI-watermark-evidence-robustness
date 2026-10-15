#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface"
exec .conda/b-attack/bin/python B/attack.py "$@"
