#!/usr/bin/env bash
ROOT=__PROJECT_ROOT__/stage10_9_three_dataset_baselines
cd "$ROOT" || exit 1
echo '===== processes ====='; pgrep -af 'stage10_9_three_dataset_baselines|runner/train.py|runner/evaluate.py' || true
echo '===== completed ====='; find runs results -name .complete -printf '%p\n' 2>/dev/null | sort
echo '===== recent logs ====='; find logs -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -5 | cut -d' ' -f2- | while read -r f; do echo "--- $f"; tail -n 3 "$f"; done
test -f STAGE109_COMPLETE && echo STAGE109_COMPLETE || echo STAGE109_RUNNING
