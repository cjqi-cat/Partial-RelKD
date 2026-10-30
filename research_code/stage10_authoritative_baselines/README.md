# Stage10：五个权威 Change Caption 基线

第一阶段只在 LEVIR-MCI 上执行严格 `geometry_quality` LOFO：训练只允许
`cloud / haze / gaussian_noise / jpeg / pixel_dropout / radiometric_shift`，完整留出
`blur / low_resolution / misregistration / mixed`。所有模型共用 calibrated-v3 在线退化、
`severity=2`、测试种子 `20260928`、Change-Agent 的 BLEU/ROUGE/CIDEr 实现和 31 条件测试。

模型：RSICCFormer、MCCFormer-D、Chg2Cap、SEN、RSCaMa。外部方法在此处统一使用
“允许退化在线增强 + caption CE”，用于比较架构在相同鲁棒训练数据下的能力；这不是逐项复刻
论文中的原始训练超参数。验证和选模不接触任何留出退化。

## SEN 初始化说明

默认配置使用官方代码支持的六通道 ImageNet ResNet-50 初始化
`resnet50_B6_imagenet`，因此论文表格中应写 **SEN-ImageNet**。若要严格复现 SEN 的
SSL4EO-S12 初始化，将官方 `rn50.pth.tar` 上传至 `assets/`，确认 conv1 是六通道，然后把
`sen_weight_path` 改成其绝对路径；两种初始化不要混报。

## 服务器运行

Clone the pinned upstream repositories listed in the repository-level
`THIRD_PARTY.md` into `vendor/chg2cap`, `vendor/sen`, and `vendor/rscama`.
Those directories are intentionally excluded from this release. Then run:

```bash
cd __PROJECT_ROOT__/stage10_authoritative_baselines
pip install -r requirements.txt
# RSCaMa 的官方自定义 Mamba 块需要 CUDA 扩展；在已有 torch/CUDA 环境中安装：
# setup.sh 会安装与 Python 3.12 / torch 2.5 / CUDA 12 / ABI=False 精确匹配的
# causal-conv1d 1.5.3.post1 与 mamba-ssm 2.2.4 官方 wheel。
python -m compileall -q runner
python runner/preflight.py \
  --config configs/rsiccformer.json --config configs/mccformer_d.json \
  --config configs/chg2cap.json --config configs/sen.json --config configs/rscama.json

for m in rsiccformer mccformer_d chg2cap sen rscama; do
  python runner/smoke_test.py --config configs/$m.json || break
done

nohup bash run_all.sh > launch.log 2>&1 &
echo $!
bash status.sh
```

首次运行会下载 torchvision 和 CLIP 的公开预训练权重。若服务器无法下载 CLIP，在联网机器缓存
`ViT-B/32` 的 `.pt` 文件并上传，然后把 `configs/rscama.json` 的 `clip_model` 改成绝对路径。

每个模型保存 31 条件 metrics、逐样本 predictions 和 `all_metrics.json`；总表为
`results/summary.md`。脚本在每个模型完成后删除 `last.pt`、保留 `best.pt`。预计峰值小于
20 GB，70 GB 空间足够。
