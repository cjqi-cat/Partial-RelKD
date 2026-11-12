# SCNet on SECOND-CC and WHU-CDC

This lightweight stage reuses the verified official SCNet wrapper in
`stage10_17_scnet_lofo`. Each dataset is independently target-supervised under
the same geometry-quality LOFO protocol. Channel statistics are computed once
from that dataset's training images only and cached in `runtime_configs/`.

```bash
cd __PROJECT_ROOT__/stage10_19_scnet_multidataset
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
results/scnet_multidataset_summary.md
```
