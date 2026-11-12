# Independent L_chg ablation

This is a deliberately small exploratory experiment. It changes exactly one Stage-1 coefficient from the original geometry-quality P3 configuration:

```text
lambda_change: 1.0 -> 0.0
```

`L_patch` remains enabled at `0.5`; every other Stage-1 and Stage-2 setting, split, allowed degradation list, seed, and checkpoint-selection rule is retained.

The reduced evaluation contains only:

- clean;
- the nine geometry-quality held-out conditions (`blur`, `low_resolution`, and `misregistration`, each on A/B/both);
- `mixed__both` as a stress reference.

## Server commands

```bash
cd __PROJECT_ROOT__/stage10_25_lchg_ablation
python -m compileall -q runner
python runner/preflight.py
bash -n run.sh
nohup bash run.sh > run.log 2>&1 &
echo $!
tail -f run.log
```

Check status:

```bash
bash status.sh
```

After completion:

```bash
cat results/summary.md
```

Interpret `P3 w/o L_chg - P3`: a negative difference means that removing `L_chg` hurts performance. This one-seed experiment is suitable for deciding whether the loss is useful, but not for a formal significance claim.
