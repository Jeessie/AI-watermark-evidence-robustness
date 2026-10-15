#!/usr/bin/env bash
# Local RTX 5090 environment; CARC P100 users should use setup_env.sh instead.
set -euo pipefail
cd "$(dirname "$0")/.."
attack_env="$PWD/.conda/b-attack"
if [[ ! -x "$attack_env/bin/python" ]]; then
    conda create --prefix "$attack_env" python=3.11 pip -y
fi
"$attack_env/bin/python" -m pip install -r B/requirements-attack.txt
"$attack_env/bin/python" -m pip check
