#!/usr/bin/env bash
pgrep -af 'stage10_18_rmnet_lofo|runner/train.py|runner/evaluate.py' || true
tail -n 20 logs/train.log 2>/dev/null || true
tail -n 20 logs/eval.log 2>/dev/null || true
test -f RMNET_COMPLETE && echo RMNET_COMPLETE || true
