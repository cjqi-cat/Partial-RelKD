#!/usr/bin/env bash
set -euo pipefail
python -m pip install -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com einops timm fvcore ptflops thop
echo SETUP_OK
