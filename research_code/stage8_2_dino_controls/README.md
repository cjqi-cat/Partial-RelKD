# Stage8.2：三数据集DINO-Clean与DINO-Aug控制实验

本阶段只补充两个同架构控制组，不修改Stage8.1及之前的任何结果：

- `DINO-Clean`：直接组合每个数据集已有的Partial-last2 encoder与clean caption teacher，评测31个条件；
- `DINO-Aug`：从完全相同的初始化出发，使用clean/允许退化视图和caption CE训练，不使用token KD、relation KD或额外clean损失。

三个数据集均采用geometry-quality LOFO。训练只允许`cloud、haze、gaussian_noise、jpeg、pixel_dropout、radiometric_shift`；`blur、low_resolution、misregistration、mixed`不得参与训练、验证和选模。测试固定`severity=2`、`degradation_seed=20260928`。

## 上传

本机PowerShell：

```powershell
scp -P <ssh-port> "<local-workspace>\stage8_2_dino_controls_server.zip" root@<server-host>:__PROJECT_ROOT__/
```

服务器：

```bash
cd __PROJECT_ROOT__
unzip -o stage8_2_dino_controls_server.zip
cd stage8_2_dino_controls

python -m compileall -q runner
python runner/make_configs.py
python runner/preflight.py
python __PROJECT_ROOT__/stage6_6_m2_backbones/runner/smoke_test.py \
  --config configs/SECOND_CC/dino_aug.json
```

看到`PREFLIGHT_OK`与`SMOKE_OK`后启动：

```bash
nohup bash run_all.sh > run.log 2>&1 &
echo $!
tail -f run.log
```

完成标志：

```text
===== DINO CONTROLS ALL DATASETS COMPLETED: ... =====
```

结果：

```text
results/LEVIR_MCI/summary.md
results/SECOND_CC/summary.md
results/WHU_CDC/summary.md
```

SECOND-CC和WHU-CDC汇总会自动连接Stage8.1已有的Change-Agent、M2和P3结果。LEVIR-MCI控制结果独立保存，后续与已有LEVIR总表合并，避免脚本误选历史实验目录。

预计新增空间2–4GB。每个DINO-Aug完成后会自动删除冗余`last.pt`，保留`best.pt`、31条件指标和逐样本预测。
