# Stage10.7 severity curves

Post-training five-level stress evaluation for Change-Agent, P3, and
P3-NoMask on LEVIR-MCI geometry-quality LOFO.  No checkpoint is retrained or
selected.  Severity 2 is byte-identical to the existing calibrated-v3 fixed
stress point; levels 1 and 3--5 are monotone extensions used only at test time.

```bash
cd __PROJECT_ROOT__/stage10_7_severity_curves
python -m compileall -q .
python runner/preflight.py
python runner/audit_severity.py --output /tmp/severity_audit.json --samples 4
# Optional model smoke test (repeat with p3 and p3_nomask):
python runner/evaluate.py --model change_agent --severity 2 --conditions blur__both --limit 2 --workers 0 --output /tmp/stage107_smoke
nohup bash run_all.sh > run.log 2>&1 &
echo $!
tail -f run.log
```

Final outputs are `results/summary.md`, `results/severity_summary.json`,
`results/condition_metrics.csv`, six PNG curves, and all paired predictions.
