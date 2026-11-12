#!/usr/bin/env bash
set -euo pipefail
cd __PROJECT_ROOT__/stage10_authoritative_baselines
pip install -r requirements.txt
python - <<'PY'
import torch
print({"torch": torch.__version__, "cuda": torch.version.cuda})
print("SETUP_OK")
PY
