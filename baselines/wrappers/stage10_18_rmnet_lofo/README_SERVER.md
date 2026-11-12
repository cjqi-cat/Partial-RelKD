# RMNet controlled geometry-quality LOFO

Official architecture: `Jiahuiqu/RMNet`. The public source hard-codes private
pretraining paths. This adapter uses the official public ResNet ImageNet branch
and requires the public VMamba-S checkpoint at:

`__WORK_ROOT__/pretrained/vssmsmall_dp03_ckpt_epoch_238.pth`

Download URL:
`https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmsmall_dp03_ckpt_epoch_238.pth`

```bash
cd __PROJECT_ROOT__/stage10_18_rmnet_lofo
bash setup.sh
nohup bash run_rmnet.sh > run.log 2>&1 &
tail -f run.log
```

Do not run SCNet and RMNet concurrently on a single GPU. Selection uses clean
validation BLEU-4; unseen and mixed test data are evaluation only.
