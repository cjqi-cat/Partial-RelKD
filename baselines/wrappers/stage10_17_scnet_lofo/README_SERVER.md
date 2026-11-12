# SCNet controlled geometry-quality LOFO

Official architecture: `sundongwei/SCNet` (MIT). This package keeps the official
ResNet101 + spatial/channel attentive encoder + difference-guided decoder and
replaces only the data, degradation, selection, and evaluation shell.

```bash
cd __PROJECT_ROOT__/stage10_17_scnet_lofo
bash setup.sh
nohup bash run_scnet.sh > run.log 2>&1 &
tail -f run.log
```

Selection is clean validation BLEU-4. Training uses only the six allowed
geometry-quality LOFO degradations. Unseen and mixed test data never select a checkpoint.
