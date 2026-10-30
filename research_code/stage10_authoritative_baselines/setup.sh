#!/usr/bin/env bash
set -euo pipefail
cd __PROJECT_ROOT__/stage10_authoritative_baselines
pip install -r requirements.txt
# The configured Aliyun index serves source archives here. Install exact
# CPython 3.12 / torch 2.5 / CUDA 12 / ABI=False release wheels instead.
WHEEL_DIR=/tmp/stage10_mamba_wheels
mkdir -p "$WHEEL_DIR"
wget -c -O "$WHEEL_DIR/causal_conv1d-1.5.3.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.3.post1/causal_conv1d-1.5.3.post1%2Bcu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
wget -c -O "$WHEEL_DIR/mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" \
  "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4%2Bcu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
pip install "$WHEEL_DIR"/*.whl
python - <<'PY'
import torch, transformers, clip
from transformers.models.mamba.modeling_mamba import is_fast_path_available
print({"torch": torch.__version__, "transformers": transformers.__version__,
       "cuda": torch.version.cuda, "mamba_fast_path": bool(is_fast_path_available)})
if not is_fast_path_available:
    raise SystemExit("RSCaMa requires Mamba CUDA fast path but it is unavailable")
print("SETUP_OK")
PY
