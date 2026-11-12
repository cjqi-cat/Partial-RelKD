#!/usr/bin/env bash
ROOT=__PROJECT_ROOT__/stage10_authoritative_baselines
pgrep -af 'stage10_authoritative_baselines|runner/(train|evaluate).py' || true
echo '===== recent ====='
tail -n 40 "$ROOT/logs/run_all.log" 2>/dev/null || true
echo '===== disk ====='
du -sh "$ROOT" 2>/dev/null || true
df -h __STORAGE_ROOT__
