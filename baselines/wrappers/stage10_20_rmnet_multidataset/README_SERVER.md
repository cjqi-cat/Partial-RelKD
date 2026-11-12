# RMNet on SECOND-CC and WHU-CDC

This lightweight stage reuses the verified official RMNet wrapper in
`stage10_18_rmnet_lofo` and the same public VMamba-S checkpoint. Each dataset
is independently target-supervised under the geometry-quality LOFO protocol.

```bash
cd __PROJECT_ROOT__/stage10_20_rmnet_multidataset
nohup bash run_all.sh > run.log 2>&1 &
echo $!
tail -f run.log
```

Status:

```bash
bash status.sh
```

Final summary:

```text
results/rmnet_multidataset_summary.md
```
