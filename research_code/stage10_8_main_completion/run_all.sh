#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=__PROJECT_ROOT__/stage10_8_main_completion;cd "$ROOT";mkdir -p logs
trap 'echo "===== FAILED line=$LINENO time=$(date "+%F %T") ====="' ERR
echo "===== START $(date '+%F %T') ====="
python -u runner/make_configs.py
python -u runner/preflight.py
for family in atmospheric noise_loss;do if [ ! -f "${family}_COMPLETE" ];then bash run_family.sh "$family" >>"logs/${family}.log" 2>&1;fi;done
for seed in 20261018 20261028;do if [ ! -f "results/multiseed/geometry_seed${seed}/.complete" ];then bash run_nomask_seed.sh "$seed" >>"logs/nomask_seed${seed}.log" 2>&1;fi;done
for ds in SECOND_CC WHU_CDC;do if [ ! -f "results/cross/$ds/p3_nomask/.complete" ];then bash run_cross_nomask.sh "$ds" >>"logs/cross_${ds}.log" 2>&1;fi;done
if [ ! -f TEST_SEEDS_COMPLETE ];then bash run_test_seeds.sh >>logs/test_seeds.log 2>&1;fi
python -u runner/efficiency.py >>logs/analysis.log 2>&1
python -u runner/qualitative.py >>logs/analysis.log 2>&1
python -u runner/summarize_all.py --root "$ROOT" >>logs/analysis.log 2>&1
touch STAGE108_COMPLETE
echo "===== MAIN COMPLETION ALL DONE $(date '+%F %T') ====="
