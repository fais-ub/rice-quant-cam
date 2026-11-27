#!/usr/bin/env bash
set -euo pipefail
VENV=${VENV:-$HOME/venvs/joiv}
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -U pip wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r "$(dirname "$0")/../requirements.txt"
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
EOF
echo "Activate with: source $VENV/bin/activate"
