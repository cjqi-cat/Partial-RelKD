# Stage10.5 P0–P2 NoMask ablation

This package reproduces the P0--P3 ablation for the independently retrained
mask-free Partial-RelKD extension.

- P0-NoMask: frozen robust partial-last2 encoder, dirty caption CE + clean-teacher token KD.
- P1-NoMask: P0 objective with end-to-end encoder updates and backbone anchoring.
- P2-NoMask: P1 plus clean-image caption CE and clean-aware validation selection.
- P3-NoMask: existing Stage10.3 result, adding relation KD.

All three models start independently from the same Stage10.3 mask-free encoder and clean-caption teacher. They are not serially initialized from one another.

Server install path:

```text
__PROJECT_ROOT__/stage10_5_p0_p2_nomask
```

Run:

```bash
cd __PROJECT_ROOT__/stage10_5_p0_p2_nomask
python -m compileall -q .
python runner/preflight.py --root "$PWD"
bash -n run_all.sh
nohup bash run_all.sh > run.log 2>&1 &
echo $!
tail -f run.log
```

Final outputs:

```text
results/summary.md
results/summary.json
results/p0_nomask/
results/p1_nomask/
results/p2_nomask/
```
